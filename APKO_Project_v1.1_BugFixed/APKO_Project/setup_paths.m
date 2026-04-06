%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  setup_paths.m
%
%  Run this once before any experiment to add all project
%  subfolders to the MATLAB path.
%
%  Usage: setup_paths()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function setup_paths()

root = fileparts(mfilename('fullpath'));

% Core algorithm files
addpath(fullfile(root, 'Algorithms'));
addpath(fullfile(root, 'Algorithms', 'PKO_Variants'));
addpath(fullfile(root, 'Algorithms', 'Competitors'));

% Initialization
addpath(fullfile(root, 'Initialization'));

% Benchmarks
addpath(fullfile(root, 'Benchmarks', 'CEC2014'));
addpath(fullfile(root, 'Benchmarks', 'CEC2017'));
addpath(fullfile(root, 'Benchmarks', 'Engineering'));

% Analysis utilities
addpath(fullfile(root, 'Analysis'));
addpath(fullfile(root, 'Utils'));

% Create Results directory if not present
results_dir = fullfile(root, 'Results');
if ~exist(results_dir, 'dir'); mkdir(results_dir); end

fprintf('[setup_paths] MATLAB path configured for APKO project.\n');
fprintf('[setup_paths] Root: %s\n', root);
fprintf('[setup_paths] IMPORTANT: Compile CEC2014 and CEC2017 MEX files:\n');
fprintf('  cd Benchmarks/CEC2014 && mex cec14_func.cpp\n');
fprintf('  cd Benchmarks/CEC2017 && mex cec17_func.cpp\n');
fprintf('[setup_paths] Ready.\n\n');
end
