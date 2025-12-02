# 📤 Google Colab Upload Instructions

## Complete Guide for Running Your EE4745 Neural Final Project in Colab

---

## 🎯 What You Need to Upload

You have **TWO OPTIONS** for running the project in Google Colab:

### ✅ **OPTION A: Use Google Drive (RECOMMENDED)**
This is the easiest and fastest method.

### ✅ **OPTION B: Upload Dataset Directly to Colab**
Works if you don't want to use Google Drive (slower for large datasets).

---

## 📋 OPTION A: Google Drive Method (Recommended)

### **Step 1: Prepare Your Google Drive**

1. **Upload the dataset to your Google Drive:**
   - Locate your `EE4745-project-data-to-release` folder on your computer
   - Upload it to your Google Drive (in the root `My Drive` or any folder)
   - **Path should be**: `/MyDrive/EE4745-project-data-to-release/`

2. **Verify dataset structure:**
   ```
   EE4745-project-data-to-release/
   ├── train/
   │   ├── baseball/
   │   ├── basketball/
   │   ├── football/
   │   ├── golf/
   │   ├── hockey/
   │   ├── rugby/
   │   ├── swimming/
   │   ├── tennis/
   │   ├── volleyball/
   │   └── weightlifting/
   └── valid/
       ├── baseball/
       ├── basketball/
       ├── football/
       ├── golf/
       ├── hockey/
       ├── rugby/
       ├── swimming/
       ├── tennis/
       ├── volleyball/
       └── weightlifting/
   ```

### **Step 2: Upload Notebook to Colab**

1. **Go to Google Colab:** https://colab.research.google.com/
2. **Upload the master notebook:**
   - Click `File` → `Upload notebook`
   - Select `EE4745_Neural_Final_Master_Colab.ipynb` from your computer
   - Or drag and drop the notebook file

3. **That's it!** The notebook will automatically:
   - Clone your GitHub repository
   - Install all dependencies
   - Link to your Google Drive dataset
   - Run all experiments

### **Step 3: Run the Notebook**

1. **Click "Runtime" → "Run all"** to execute everything
2. **Or run cell by cell** to monitor progress
3. **When prompted**, click "Connect to Google Drive" and authorize access
4. **Wait for completion** (2-3 hours depending on epochs)
5. **Download results** when finished

---

## 📋 OPTION B: Direct Upload Method

### **Step 1: Prepare Dataset ZIP**

1. **Compress your dataset:**
   ```bash
   # On Mac/Linux
   zip -r EE4745-project-data-to-release.zip EE4745-project-data-to-release/

   # On Windows (right-click folder → Send to → Compressed folder)
   ```

2. **Verify ZIP file** contains both `train/` and `valid/` folders

### **Step 2: Upload to Colab**

1. **Go to Google Colab:** https://colab.research.google.com/
2. **Upload the master notebook:**
   - Click `File` → `Upload notebook`
   - Select `EE4745_Neural_Final_Master_Colab.ipynb`

3. **Run the notebook cells until you reach** "OPTION B: Upload Dataset Directly"
4. **Run that cell** - it will prompt you to upload the ZIP file
5. **Select your ZIP file** and wait for upload (may take 10-15 minutes for ~1GB)
6. **Continue running** the rest of the cells

---

## 📂 Files You Need (Summary)

### **Required:**
- ✅ `EE4745_Neural_Final_Master_Colab.ipynb` (the notebook you'll upload to Colab)
- ✅ Dataset: `EE4745-project-data-to-release/` (either in Google Drive OR as ZIP)

### **Automatically Handled:**
- ✅ Source code (cloned from GitHub)
- ✅ Dependencies (installed by notebook)
- ✅ All training scripts (in the repository)

---

## ⚙️ Colab Configuration Tips

### **Adjust Training Time**

If you want faster results for testing, edit these parameters in the notebook:

```python
# In the training cells, change:
--epochs 20    # Instead of 50 (faster)
--epochs 5     # For very quick testing
```

### **GPU vs CPU**

The project is designed for CPU, but you can use GPU for faster training:

1. **Enable GPU:** `Runtime` → `Change runtime type` → `Hardware accelerator` → `GPU`
2. **In training cells, change:**
   ```python
   --device cpu    # Change to: --device cuda
   ```

### **Memory Management**

If you run out of memory:

```python
# Reduce batch sizes in training cells:
--batch_size 32    # Change to: --batch_size 16
--batch_size 16    # Change to: --batch_size 8
```

---

## 🚀 Step-by-Step Execution Guide

### **Phase 1: Initial Setup (5 minutes)**
1. Upload notebook to Colab
2. Mount Google Drive (or prepare ZIP upload)
3. Run setup cells
4. Verify dataset is accessible

### **Phase 2: Problem A (45-90 minutes)**
1. Train SimpleCNN model
2. Train ResNetSmall model
3. Generate interpretability analysis
4. Review training curves and results

### **Phase 3: Problem B (20-40 minutes)**
1. Generate adversarial examples
2. Run transferability analysis
3. Create interpretability comparisons
4. Review attack effectiveness

### **Phase 4: Problem C (30-60 minutes)**
1. Apply unstructured pruning
2. Fine-tune pruned models
3. Measure performance metrics
4. Analyze trade-offs

### **Phase 5: Final Compilation (10 minutes)**
1. Generate master dashboard
2. Compile all results
3. Download results archive
4. Review executive summary

### **Total Time: 2-3 hours** (depending on configuration)

---

## 📥 Downloading Results

### **At the End of the Notebook:**

The notebook will automatically create a ZIP archive containing:
- `results/` - All experimental results
- `checkpoints/` - All trained models
- `logs/` - TensorBoard training logs
- `notebooks/` - Analysis notebooks

### **Download Options:**

1. **Automatic download** at the end of the notebook
2. **Manual download** from Colab file browser:
   - Click folder icon on left sidebar
   - Right-click on ZIP file
   - Select "Download"
3. **Save to Google Drive** (optional):
   ```python
   # Add this cell to save to Drive:
   !cp EE4745_Neural_Final_Results_*.zip /content/drive/MyDrive/
   ```

---

## 🐛 Troubleshooting

### **Dataset Not Found**

```
❌ Error: Dataset not found at: /content/drive/MyDrive/EE4745-project-data-to-release/
```

**Solution:**
1. Check your Google Drive path
2. Update the path in the notebook:
   ```python
   drive_dataset_path = '/content/drive/MyDrive/YOUR_FOLDER_NAME/EE4745-project-data-to-release'
   ```
3. Or use OPTION B (direct upload)

### **Out of Memory**

```
❌ RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce batch size:
   ```python
   --batch_size 16  # Or even 8
   ```
2. Use CPU instead of GPU:
   ```python
   --device cpu
   ```

### **Session Timeout**

```
⚠️ Colab session disconnected
```

**Solution:**
1. Colab has a maximum session time
2. Run in shorter segments if needed
3. Save intermediate results:
   ```python
   # After each problem, save to Drive:
   !cp -r results/ /content/drive/MyDrive/results_backup/
   ```

### **GitHub Clone Fails**

```
❌ fatal: destination path 'Neural-Final' already exists
```

**Solution:**
1. Remove existing directory:
   ```python
   !rm -rf Neural-Final
   !git clone https://github.com/Tyler-Trauernicht/Neural-Final.git
   ```

---

## ✅ Pre-Flight Checklist

Before running the notebook, make sure:

- [ ] You have a Google account
- [ ] Dataset is uploaded to Google Drive OR prepared as ZIP
- [ ] You have `EE4745_Neural_Final_Master_Colab.ipynb` file
- [ ] You have 2-3 hours of free time
- [ ] You have stable internet connection
- [ ] You've saved the notebook to your Google Drive (optional)

---

## 📊 Expected Outputs

After completion, you'll have:

### **Trained Models:**
- `SimpleCNN-original.pt` (~2.4 MB)
- `ResNetSmall-original.pt` (~10.6 MB)
- Pruned models at 20%, 50%, 80% sparsity

### **Results:**
- Training curves (PNG)
- Confusion matrices (PNG)
- 100+ interpretability visualizations
- Attack analysis reports
- Pruning trade-off plots
- Master performance dashboard

### **Reports:**
- Executive summary (MD)
- Technical analysis reports (MD)
- Performance comparison tables (CSV, LaTeX)

---

## 🎯 Quick Start Command

**For the impatient:**

1. Upload notebook to Colab
2. Run this single cell to do everything:
```python
# This will take 2-3 hours
%run EE4745_Neural_Final_Master_Colab.ipynb
```

**But we recommend** running cell-by-cell to monitor progress!

---

## 💡 Pro Tips

### **Save Time:**
- Use fewer epochs for initial testing (5-10)
- Use GPU if available (`Runtime` → `Change runtime type`)
- Save intermediate results to Google Drive

### **Monitor Progress:**
- Run cells individually to see progress
- Check TensorBoard in separate tab
- Keep an eye on memory usage

### **Backup Strategy:**
- Periodically save results to Google Drive
- Download checkpoints after each problem
- Keep the results ZIP safe

---

## 📞 Need Help?

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Review notebook output for error messages**
3. **Verify dataset structure and paths**
4. **Try reducing epochs/batch sizes for testing**
5. **Consult the main README.md for detailed documentation**

---

## 🎉 You're Ready!

You now have everything you need to run the complete EE4745 Neural Final Project in Google Colab!

**Next Steps:**
1. Upload `EE4745_Neural_Final_Master_Colab.ipynb` to Google Colab
2. Ensure dataset is accessible (Google Drive or ZIP)
3. Run the notebook cell by cell
4. Download results when complete
5. Write your final report using the generated materials

**Good luck! 🚀**