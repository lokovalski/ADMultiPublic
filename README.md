CODE OVERVIEW (sorry in advance)
------------------------
- dataloader.py: loads data and whatnot
- unimodal_baseline.py: makes TabularBaseline and MRIBaseline objects. need the dataloaders to run
- experiment.py: contains code for fusion models
- fig.py: defunct plotting code which I don't think I ended up using
- (!!!) all the notebooks (i.e. diary.ipynb, tabtest.ipynb, mritest.ipynb, fusiontest.ipynb): disgusting messes of me running stuff and getting figures. there will be repeat methods because my kernel would randomly break and updates to say unimodal_baseline.py wouldn't import so I'd copy the whole class just to keep running without retraining. TLDR: these are actually abhorrent and I apologize to anyone who might venture into these files but the bulk of results are here.
    - fusiontest.ipynb also currently is meant to run the ablation study. if you want to try running the regular version, command F and delete every np.delete(...)

NOTE: the data is not included here as (1) it is 30GB and (2) the application I made to gain access to it said no sharing.

OASIS-1 Dataset Summary
------------------------

General Overview:
- The dataset includes 416 adult participants ranging from 18 to 96 years old.
- Of these, 100 participants over age 60 have been clinically diagnosed with very mild to mild Alzheimer’s Disease (AD).
- Each subject typically has 3 to 4 T1-weighted MRI scans acquired during a single session.
- Data is ~15.8 GB compressed, ~50 GB uncompressed

MRI Imaging Data:
- Raw scans consist of multiple repetitions to improve signal-to-noise ratio.
- Processed outputs include:
  - A motion-corrected average image
  - An atlas-registered, gain-field corrected image (whatever that means?)
  - A brain-masked image with non-brain voxels set to zero
- All scans are in Analyze 7.5 format (.hdr/.img), 16-bit, big-endian encoded.

Directory Structure:
- There are 12 discs, containing ~40 subjects each
- For each subject, there is:
  - An .xml metadata file and a corresponding .txt version
  - A RAW/ directory with multiple scan repetitions (e.g., mpr-1, mpr-2)
  - A PROCESSED/ directory containing:
    - SUBJ_111/: subject-space averaged MRI
    - T88_111/: bias-corrected, atlas-aligned (1mm³ voxel) MRI images
    - t4_files/: transformation matrices used in registration
  - FSL_SEG/: tissue segmentations (gray, white, CSF)

Tabular Data:
- Provided in oasis_cross-sectional.csv and includes:
  - Demographics: age, sex, education, socioeconomic status (SES)
  - Cognitive scores: MMSE (Mini-Mental State Exam), CDR (Clinical Dementia Rating)
    - CDR scores: 0 (normal), 0.5 (very mild), 1 (mild), 2 (moderate)
  - MRI-derived anatomical metrics:
    - eTIV: estimated total intracranial volume
    - ASF: atlas scaling factor
    - nWBV: normalized whole brain volume

Best Practices:
- For deep learning tasks, use the T88_111 images:
  - These are bias-field corrected, gain-field corrected, and registered to a common atlas
  - They have isotropic voxel resolution (1.0 × 1.0 × 1.0 mm) and consistent orientation across subjects

