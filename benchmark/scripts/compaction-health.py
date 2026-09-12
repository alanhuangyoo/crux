import json,glob,os,sys,collections,statistics
for root in sys.argv[1:]:
    ds=sorted(glob.glob(root+"/*/"))
    if not ds: print("%-12s no run dir yet"%os.path.basename(root)); continue
    d=ds[0]; ok=bad=0; msgs=collections.Counter(); trials=0
    turns=trunc=errs=0; maxctx=[]
    for x in sorted(glob.glob(d+"*/")):
        f=os.path.join(x,"agent","pi.txt")
        if not os.path.exists(f): continue
        trials+=1; peak=0
        for line in open(f,errors="ignore"):
            try: ev=json.loads(line)
            except: continue
            if ev.get("type")=="compaction_end":
                e=ev.get("errorMessage")
                if e: bad+=1; msgs[str(e)[:62]]+=1
                else: ok+=1
                continue
            # message_start / message_end / turn_end each carry the same
            # assistant message, so anything counted per-message has to pick one
            # of them -- counting all three inflated every turn count by 3x.
            if ev.get("type")!="message_end": continue
            m=ev.get("message") or {}
            if m.get("role")=="assistant":
                turns+=1; sr=m.get("stopReason")
                if sr=="length": trunc+=1
                elif sr=="error": errs+=1
                peak=max(peak,(m.get("usage") or {}).get("totalTokens") or 0)
        if peak: maxctx.append(peak)
    tot=ok+bad
    print("%-12s trials=%-3d turns=%-5d | compaction %-4d ok=%-4d fail=%-4d (%s) | trunc=%s err=%s | peak ctx med=%s" % (
        os.path.basename(root),trials,turns,tot,ok,bad,
        ("%.0f%%"%(ok/tot*100)) if tot else "n/a",
        ("%.1f%%"%(trunc/turns*100)) if turns else "n/a",
        ("%.1f%%"%(errs/turns*100)) if turns else "n/a",
        ("%d"%statistics.median(maxctx)) if maxctx else "n/a"))
    for m,c in msgs.most_common(3): print("      %5d  %s"%(c,m))
