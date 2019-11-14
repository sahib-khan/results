% This script can be used to perform a comparative analyis of the experiments
% in the same manner as for the VOT challenge
% You can copy and modify it to create a different analyis

addpath('/root/collab/vot-toolkit'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

%error('Analysis not configured! Please edit run_analysis.m file.'); % Remove this line after proper configuration

trackers = tracker_list('ADT', 'DensSiam','SiamFC','SiamRPN','KCF','DSiam','CCOT'); % TODO: add more trackers here

workspace_analyze(trackers, sequences, experiments, 'report_ADT', 'Title', 'Report for vot2018');

