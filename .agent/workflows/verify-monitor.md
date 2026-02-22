---
description: How to verify that the training monitor is functional
---

After any code update to the training pipeline, follow these steps to ensure the monitor remains functional:

1. **Launch Training**: Start the training script (e.g., `python primal_train_modular.py`).
2. **Check Log Flow**: Verify that training output is appearing in `training_v5_6.log`.
   - Command: `tail -f training_v5_6.log`
3. **Check Stat Updates**: Verify that `stats_coder.json` is being updated with recent timestamps and step numbers.
   - Command: `tail -n 1 stats_coder.json`
4. **Verify Dashboard Server**: Ensure the `trinity_peak.py` server is running and serving API requests.
   - Test Command: `curl http://localhost:8000/api/stats/project_real`
5. **UI Validation**: Open `dashboard/index.html` in a browser and confirm that charts are plotting new points.
