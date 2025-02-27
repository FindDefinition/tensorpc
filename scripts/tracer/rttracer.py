from tensorpc.dbg.rttracer import fetch_trace_result
import gzip

def __main():
    import json 
    with open("debug.gz", "wb") as f:
        f.write(fetch_trace_result("10.21.99.122:57001", "debug"))
    # print(fetch_trace_result("10.21.102.72:53651", "debug"))

def __main2():
    import io 
    import zipfile
    from tensorpc.flow.components.plus.dbg.dbgpanel import list_all_dbg_server_in_machine
    proc_metas = list_all_dbg_server_in_machine()
    res_all = []
    _use_perfetto_undoc_zip_of_gzip = False
    zip_ss = io.BytesIO()
    zip_mode = zipfile.ZIP_DEFLATED if not _use_perfetto_undoc_zip_of_gzip else zipfile.ZIP_STORED
    compresslevel = 9 if not _use_perfetto_undoc_zip_of_gzip else None

    with zipfile.ZipFile(zip_ss, mode="w", compression=zip_mode, compresslevel=compresslevel) as zf:
        for i, meta in enumerate(proc_metas):
            res = fetch_trace_result(f"localhost:{meta.port}", "debug") 
            zf.writestr(f"{i}.json", gzip.decompress(res))
    with open("debug.zip", "wb") as f:
        f.write(zip_ss.getvalue())

def __main3():
    import io 
    import zipfile
    import json 
    res_all = []
    res_proc_names = []
    with zipfile.ZipFile("debug.zip", mode="r") as zf:
        for name in zf.namelist():
            res_bytes = zf.read(name)
            res = json.loads(res_bytes)
            res_all.append((res, res["traceEvents"][0]["args"]["name"]))
    res_all.sort(key=lambda x: x[1])
    res_all = [x[0] for x in res_all]
    compare_inds = [2, 3]
    print(res_all[compare_inds[0]]["traceEvents"][0])
    print(res_all[compare_inds[1]]["traceEvents"][0])
    lfs = res_all[compare_inds[0]]["traceEvents"][2:] # ignore meta events
    rfs = res_all[compare_inds[1]]["traceEvents"][2:] # ignore meta events
    lfs.sort(key=lambda x: x["ts"])
    rfs.sort(key=lambda x: x["ts"])
    cnt = 0
    history_length = 10
    for lfs_item, rfs_item in zip(lfs, rfs):
        name_lfs = lfs_item["name"]
        name_rfs = rfs_item["name"]
        # if "einops" in name_lfs:
        #     continue
        if name_lfs != name_rfs:
            print("WTF", cnt, name_lfs, name_rfs)
            for j in range(max(0, cnt - history_length), min(len(lfs), cnt + 2)):
                name_parts = lfs[j]["name"].split("(")
                name = f"{name_parts[0]} ({name_parts[1]}"
                print(name)
            print("--------------------")
            for j in range(max(0, cnt - history_length), min(len(rfs), cnt + 2)):
                name_parts = rfs[j]["name"].split("(")
                name = f"{name_parts[0]} ({name_parts[1]}"

                print(name)

            # print("DIFFERENT NAMES", lfs_item, rfs_item)
            breakpoint()
        elif lfs_item["ph"] == "I":
            print("INSTANT", lfs_item["name"], lfs_item["args"], rfs_item["args"])
        # print(lfs_item, rfs_item)
        # breakpoint()
        cnt += 1

if __name__ == "__main__":
    __main3()