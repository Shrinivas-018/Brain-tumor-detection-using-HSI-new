import csv
rows = list(csv.DictReader(open('models/training_log.csv')))
if not rows:
    print("No epoch data yet.")
else:
    dices = [float(r['val_dice']) for r in rows]
    senss = [float(r['val_sens']) for r in rows]
    best_dice = max(dices)
    best_sens = max(senss)
    print(f"Epochs completed: {len(rows)}")
    print(f"Best val_dice:    {best_dice:.4f}")
    print(f"Best val_sens:    {best_sens:.4f}")
    print(f"Train loss:       {float(rows[0]['train_loss']):.4f} -> {float(rows[-1]['train_loss']):.4f}")
    print()
    for r in rows:
        d = float(r['val_dice'])
        s = float(r['val_sens'])
        tag = " *** BEST" if d == best_dice and d > 0 else ""
        print(f"  Ep{int(r['epoch']):>3}: loss={float(r['train_loss']):.4f} | val_dice={d:.4f} | sens={s:.4f} | lr={float(r['lr']):.1e}{tag}")
