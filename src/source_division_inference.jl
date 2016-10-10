import WCS

using Base.Threads


function fetch_catalog(rcf, stagedir)
    # note: this call to read_photoobj_files considers only primary detections.
    catalog = SDSSIO.read_photoobj_files([rcf,], stagedir)

    # we're ignoring really faint sources entirely...not even using them to
    # render the background
    isnt_faint(entry) = maximum(entry.star_fluxes) >= MIN_FLUX
    filter(isnt_faint, catalog)
end


@inline function in_box(entry::CatalogEntry, box::BoundingBox)
    box.ramin < entry.pos[1] < box.ramax &&
        box.decmin < entry.pos[2] < box.decmax
end


function count_sources(rcfs::Vector{RunCamcolField}, box::BoundingBox, stagedir::String)
    if nodeid == 1
        nputs(nodeid, "$(length(rcfs)) RCFs")
    end

    # RCF index, catalog index
    tasks = Vector{Tuple{Int64,Int64}}()
    for i = 1:length(rcfs)
        cat_entries = fetch_catalog(rcfs[i], stagedir)
        for j = 1:length(cat_entries)
            if in_box(cat_entries[j], box)
                push!(tasks, (i,j))
            end
        end
    end

    if nodeid == 1
        nputs(nodeid, "$(length(tasks)) tasks")
    end

    sync()

    return tasks
end


function clean_cache(cache::Dict)
    lru_rcf = RunCamcolField(0, 0, 0)
    lru_interval = 0.0
    curr_time = time()
    for (rcf, (imgs, cat, lru)) in cache
        use_interval = curr_time - lru
        if use_interval > lru_interval
            lru_rcf = rcf
            lru_interval = use_interval
        end
    end
    ntputs(nodeid, threadid(), "discarding $(lru_rcf.run), $(lru_rcf.camcol), $(lru_rcf.field)")
    delete!(cache, lru_rcf)
end


function optimize_source(taskidx::Int64, tasks::Vector{Tuple{Int64,Int64}},
                         rcfs::Vector{RunCamcolField},
                         cache::Dict, cache_lock::SpinLock,
                         stagedir::String, times::InferTiming)
    tid = threadid()

    images = Vector{TiledImage}()

    rcf_idx, cat_idx = tasks[taskidx]
    rcf = rcfs[rcf_idx]
    #ntputs(nodeid, tid, "loading catalog for task $taskidx")
    tic()
    catalog = fetch_catalog(rcf, stagedir)
    times.load_cat = times.load_cat + toq()
    entry = catalog[cat_idx]

    #ntputs(nodeid, tid, "getting overlapping fields for task $taskidx")
    t_box = BoundingBox(entry.pos[1] - 1e-8, entry.pos[1] + 1e-8,
                        entry.pos[2] - 1e-8, entry.pos[2] + 1e-8)
    surrounding_rcfs = get_overlapping_fields(t_box, stagedir)

    #tic()
    for srcf in surrounding_rcfs
        lock(cache_lock)
        cached_imgs, cached_cat, _ = get(cache, srcf) do
            ntputs(nodeid, tid, "loading $(srcf.run), $(srcf.camcol), $(srcf.field)")
            tic()
            field_images = SDSSIO.load_field_images(srcf, stagedir)
            times.load_img = times.load_img + toq()
            tiled_images = [TiledImage(img) for img in field_images]

            neighbors = Vector{CatalogEntry}()
            if srcf.run != rcf.run ||
                    srcf.camcol != rcf.camcol ||
                    srcf.field != rcf.field
                tic()
                append!(neighbors, fetch_catalog(srcf, stagedir))
                times.load_cat = times.load_cat + toq()
            end
            tiled_images, neighbors, time()
        end
        push!(cache, srcf => (cached_imgs, cached_cat, time()))
        if length(cache) > 50
            clean_cache(cache)
        end
        unlock(cache_lock)
        append!(images, cached_imgs)
        append!(catalog, cached_cat)
    end
    #ntputs(nodeid, tid, "fetched data to infer $(entry.objid) in $(toq()) secs")

    neighbor_indexes = Infer.find_neighbors([cat_idx,], catalog, images)[1]
    neighbors = catalog[neighbor_indexes]

    gc_stats = Base.gc_num()
    elapsed_time = time_ns()

    t0 = time()
    vs_opt = Infer.infer_source(images, neighbors, entry)
    runtime = time() - t0

    elapsed_time = time_ns() - elapsed_time
    gc_diff_stats = Base.GC_Diff(Base.gc_num(), gc_stats)
    ntputs(nodeid, tid, "$(entry.objid): ",
           time_report_str(elapsed_time, gc_diff_stats.allocd,
                           gc_diff_stats.total_time,
                           Base.gc_alloc_count(gc_diff_stats)))
    #ntputs(nodeid, tid, "$(entry.objid): $(runtime) secs")

    InferResult(entry.thing_id, entry.objid, entry.pos[1], entry.pos[2],
                vs_opt, runtime)
end


function optimize_sources(tasks::Vector{Tuple{Int64,Int64}},
                          rcfs::Vector{RunCamcolField},
                          stagedir::String,
                          timing::InferTiming)
    num_work_items = length(tasks)

    # inference results
    results = Vector{InferResult}()
    results_lock = SpinLock()

    cache = Dict{RunCamcolField,
                 Tuple{Vector{TiledImage},Vector{CatalogEntry},Float64}}()
    cache_lock = SpinLock()

    # per-thread timing
    ttimes = Array(InferTiming, nthreads())

    # create Dtree and get the initial allocation
    dt, isparent = DtreeScheduler(num_work_items, 0.4, ceil(Int64, nthreads() / 4))
    numwi, (startwi, endwi) = initwork(dt)
    rundt = runtree(dt)

    nputs(nodeid, "dtree: initial work: $numwi ($startwi-$endwi)")

    widx = 1
    wilock = SpinLock()

    #gc_freq = 3
    #gc_ctr = 1

    function process_tasks()
        tid = threadid()
        ttimes[tid] = InferTiming()
        times = ttimes[tid]

        if rundt && tid == 1
            ntputs(nodeid, tid, "dtree: running tree")
            while runtree(dt)
                cpu_pause()
            end
        else
            while true
                tic()
                lock(wilock)
                if endwi == 0
                    ntputs(nodeid, tid, "dtree: out of work")
                    unlock(wilock)
                    times.sched_ovh = times.sched_ovh + toq()
                    break
                end
                if widx > numwi
                    ntputs(nodeid, tid, "dtree: getting work")
                    numwi, (startwi, endwi) = getwork(dt)
                    ntputs(nodeid, tid, "dtree: $numwi work items ($startwi-$endwi)")
                    if endwi > 0
                        widx = 1
                    end
                    unlock(wilock)
                    times.sched_ovh = times.sched_ovh + toq()
                    continue
                end
                taskidx = startwi + widx - 1
                widx = widx + 1
                unlock(wilock)
                times.sched_ovh = times.sched_ovh + toq()
                #ntputs(nodeid, tid, "running task $taskidx")

                result = InferResult(0, "", 0.0, 0.0, [0.0], 1.0)
                tries = 1
                while tries <= 3
                    result = try
                        optimize_source(taskidx, tasks, rcfs,
                                        cache, cache_lock,
                                        stagedir, times)
                    catch exc
                        ntputs(nodeid, tid, "$exc running task $taskidx on try $tries")
                        tries = tries + 1
                        continue
                    end
                    break
                end
                if tries > 3
                    ntputs(nodeid, tid, "exception running task $taskidx on 3 tries, giving up")
                    continue
                end

                #ntputs(nodeid, tid, "completed task $taskidx ($(result.objid))")
                lock(results_lock)
                push!(results, result)
                unlock(results_lock)

                #if tid == 1
                #    if gc_ctr == gc_freq
                #        nputs(nodeid, "running GC")
                #        gc_ctr = 1
                #        gc_enable(true)
                #        gc(false)
                #        gc_enable(false)
                #    end
                #    gc_ctr = gc_ctr + 1
                #end
            end
        end
    end

    #gc_enable(false)
    tic()
    ccall(:jl_threading_run, Void, (Any,), Core.svec(process_tasks))
    #process_tasks()
    #ccall(:jl_threading_profile, Void, ())
    timing.opt_srcs = toq()
    #gc_enable(true)
    #gc()

    if nodeid == 1
        nputs(nodeid, "complete")
    end
    tic()
    finalize(dt)
    timing.wait_done = toq()

    for tt in ttimes
        add_timing!(timing, tt)
    end

    return results
end


function set_thread_affinity(nid::Int, ppn::Int, tid::Int, nthreads::Int, show_cpu::Bool)
    cpu = (((nid - 1) % ppn) * nthreads)
    show_cpu && ntputs(nid, tid, "bound to $(cpu + tid)")
    mask = zeros(UInt8, 4096)
    mask[cpu + tid] = 1
    uvtid = ccall(:uv_thread_self, UInt64, ())
    ccall(:uv_thread_setaffinity, Int, (Ptr{Void}, Ptr{Void}, Ptr{Void}, Int64),
          pointer_from_objref(uvtid), mask, C_NULL, 4096)
end


function affinitize()
    ppn = try
        parse(Int, ENV["JULIA_EXCLUSIVE"])
    catch exc
        return
    end
    show_cpu = try
        parse(Bool, ENV["CELESTE_SHOW_AFFINITY"])
    catch exc
        false
    end
    function threadfun()
        set_thread_affinity(nodeid, ppn, threadid(), nthreads(), show_cpu)
    end
    ccall(:jl_threading_run, Void, (Any,), Core.svec(threadfun))
end


"""
Fit the Celeste model to sources in a given ra, dec range,
based on data from specified fields
- box: a bounding box specifying a region of sky
"""
function divide_sources_and_infer(
                box::BoundingBox,
                stagedir::String;
                timing=InferTiming(),
                outdir=".")
    affinitize()

    # read the run-camcol-field triplets for this box
    rcfs = get_overlapping_fields(box, stagedir)

    # get the sources we must process for this box
    tasks = count_sources(rcfs, box, stagedir)

    # process the sources
    if length(tasks) > 0
        results = optimize_sources(tasks, rcfs, stagedir, timing)
        timing.num_srcs = length(results)

        tic()
        save_results(outdir, box, results)
        timing.write_results = toq()
    end

    finalize(tasks)
end

