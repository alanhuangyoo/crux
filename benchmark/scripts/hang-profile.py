import json,glob,os,sys,collections,pathlib
for arm in sys.argv[1:]:
    run=sorted(pathlib.Path(arm).iterdir())[0]
    ev=list(json.loads((run/"result.json").read_text())["stats"]["evals"].values())[0]
    timed=set(ev.get("exception_stats",{}).get("AgentTimeoutError",[]))
    kinds=collections.Counter(); detail=[]
    for d in sorted(glob.glob(str(run)+"/*/")):
        name=os.path.basename(d.rstrip("/"))
        if name not in timed: continue
        f=os.path.join(d,"agent","pi.txt")
        if not os.path.exists(f): continue
        last=[]
        for line in open(f,errors="ignore"):
            try: e=json.loads(line)
            except: continue
            last.append(e)
        if not last: continue
        tail=[str(e.get("type")) for e in last[-6:]]
        final=tail[-1]
        # what was left open at the end
        opened=None
        if final in ("tool_execution_start","tool_execution_update"):
            opened="tool never returned"
            args=""
            for e in reversed(last):
                m=e.get("message") or {}
                for b in (m.get("content") or []) if isinstance(m.get("content"),list) else []:
                    if isinstance(b,dict) and b.get("type")=="toolCall":
                        args=json.dumps(b.get("arguments"),default=str)[:150]; break
                if args: break
            detail.append((name.split("__")[0],opened,args))
        elif final in ("message_start","turn_start"):
            opened="model request never returned"
            detail.append((name.split("__")[0],opened,""))
        else:
            opened="ended cleanly at "+final
            detail.append((name.split("__")[0],opened,""))
        kinds[opened.split(" at ")[0]]+=1
    print("=== %s  (%d timed out)" % (os.path.basename(arm),len(timed)))
    for k,v in kinds.most_common(): print("   %-32s %d" % (k,v))
    for n,k,a in detail:
        print("     %-30s %-30s %s" % (n,k,a))
