FIXIT Please join it with Systemdocs.

demoCompareForecasts.m - computes forecast using Dudec's data for 3 years. Active methods
are VAR, Neural Net and SVR. Due to too many features, VAR is unstable and it's forecast is
not presented on plot.

directories:
feature_generation/ feature generation modules
frc/ forecasting modules
utils/ plots, data cleaning and other supplementary code
tmp/ stores files that should potentially be removed.
data/ each dataset should be placed in a separate folder 'DataName' inside data/ directory.
A dataset is accompanied by a loader 'LoadDataName', stored in load_data/


demoGeneratingFeatures.m - creates new structures that contain new features. Using a 
pseudocolor plot to present regression matices.

draft.m - it's a draft scroll of code, not for presentation purposes.