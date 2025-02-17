   
% Create corresponding labels for each signal
labels = {'1 T', '2 T'};
signal_files = {'signal1.csv', 'signal2.csv'} % signal data
bgnd_file = 'bgnd.csv' % background data
ref_file = 'ref.csv' % reference data

min_freq = 0   % Unit: THz
max_freq = 1   % Unit: THz

% Call the function to clean the signals and plot the FFT spectra
spectrum_clean_signal(signal_files, bgnd_file, labels, min_freq, max_freq);
normalized_spectrum(signal_files, bgnd_file, ref_file, labels, min_freq, max_freq)

function normalized_spectrum(signal_files, bgnd_file, ref_file, labels, min_freq, max_freq)
% spectrum_clean_signal - Plots the FFT spectrum of multiple signals from files.
% If bgnd_file is provided, it subtracts the background from each signal.
% If ref_file is provided, it divides the result by the reference.
%
% Syntax: spectrum_clean_signal(signal_files, bgnd_file, ref_file, labels)
%
% Inputs:
%    signal_files - A cell array of file names for the signals (CSV format)
%    bgnd_file    - A single file name for the background signal (CSV format) or empty []
%    ref_file     - A single file name for the reference signal (CSV format) or empty []
%    labels       - A cell array of labels (strings) for each signal
%
% Example:
%    signal_files = {'signal1.csv', 'signal2.csv'};
%    bgnd_file = 'bgnd.csv';
%    ref_file = 'ref.csv';
%    labels = {'Signal 1', 'Signal 2'};
%    spectrum_clean_signal(signal_files, bgnd_file, ref_file, labels);
%
% Assumptions:
%    - Each CSV file has two columns: time and signal.
%    - The first row is a header and is ignored.
%    - All signals, background, and reference have the same time vector.

    % Check that signal_files and labels are provided as cell arrays
    if ~(iscell(signal_files) && iscell(labels))
        error('signal_files and labels must be provided as cell arrays.');
    end

    nSignals = length(signal_files);
    if length(labels) ~= nSignals
        error('Number of signal files and labels must be the same.');
    end

    % Initialize cell arrays to store time and signal data
    time = [];
    signals = cell(1, nSignals);

    % Read each signal file
    for k = 1:nSignals
        [time_k, signal_k] = read_time_signal_csv(signal_files{k});
        
        % If it's the first file, store the time vector
        if isempty(time)
            time = time_k;
        else
            % Check if time vectors are consistent
            if ~isequal(time, time_k)
                error('Time vectors are not consistent across all files.');
            end
        end
        
        % Store the signal
        signals{k} = signal_k;
    end

    % Read the background signal if provided
    if ~isempty(bgnd_file)
        [time_bgnd, bgnd] = read_time_signal_csv(bgnd_file);
        % Check if background time vector is consistent
        if ~isequal(time, time_bgnd)
            error('Time vector in background file is not consistent with signal files.');
        end
    else
        bgnd = [];  % No background provided
    end

    % Read the reference signal if provided
    if ~isempty(ref_file)
        [time_ref, ref] = read_time_signal_csv(ref_file);
        % Check if reference time vector is consistent
        if ~isequal(time, time_ref)
            error('Time vector in reference file is not consistent with signal files.');
        end
    else
        ref = [];  % No reference provided
    end

    figure;
    hold on;

    % Loop through each signal
    for k = 1:nSignals
        % Process the signal
        processed_signal = signals{k};
        
        % If background is provided, subtract it
        if ~isempty(bgnd)
            processed_signal = processed_signal - bgnd;
        end
        
        % If reference is provided, divide by it
        if ~isempty(ref)
            processed_signal = processed_signal ./ ref;
        end
        
        % Compute FFT (assuming computeFFT returns freq in THz)
        [freq, fft_data] = computeFFT(time, processed_signal);
        % Plot the magnitude spectrum
        plot(freq, abs(fft_data), 'LineWidth', 1.5);
    end

    hold off;
    xlabel('Frequency (THz)');
    ylabel('Magnitude');
    title('Normalized Spectrum');
    grid on;
    % Limit x-axis to half of the maximum frequency for clarity
    xlim([min_freq, max_freq]);
    % Add legend with provided labels
    legend(labels, 'Location', 'Best');
end


function spectrum_clean_signal(signal_files, bgnd_file, labels, min_freq, max_freq)
% spectrum_clean_signal - Plots the FFT spectrum of multiple signals from files.
% If bgnd_file is provided, it subtracts the background from each signal before computing the FFT.
%
% Syntax: spectrum_clean_signal(signal_files, bgnd_file, labels)
%
% Inputs:
%    signal_files - A cell array of file names for the signals (CSV format)
%    bgnd_file     - A single file name for the background signal (CSV format) or empty []
%    labels       - A cell array of labels (strings) for each signal
%
% Example:
%    signal_files = {'signal1.csv', 'signal2.csv'};
%    bgnd_file = 'bgnd.csv';
%    labels = {'Signal 1', 'Signal 2'};
%    spectrum_clean_signal(signal_files, bgnd_file, labels);
%
% Assumptions:
%    - Each CSV file has two columns: time and signal.
%    - The first row is a header and is ignored.
%    - All signals and the background have the same time vector.

    % Check that signal_files and labels are provided as cell arrays
    if ~(iscell(signal_files) && iscell(labels))
        error('signal_files and labels must be provided as cell arrays.');
    end

    nSignals = length(signal_files);
    if length(labels) ~= nSignals
        error('Number of signal files and labels must be the same.');
    end

    % Initialize cell arrays to store time and signal data
    time = [];
    signals = cell(1, nSignals);

    % Read each signal file
    for k = 1:nSignals
        [time_k, signal_k] = read_time_signal_csv(signal_files{k});
        
        % If it's the first file, store the time vector
        if isempty(time)
            time = time_k;
        else
            % Check if time vectors are consistent
            if ~isequal(time, time_k)
                error('Time vectors are not consistent across all files.');
            end
        end
        
        % Store the signal
        signals{k} = signal_k;
    end

    % Read the background signal if provided
    if ~isempty(bgnd_file)
        [time_bgnd, bgnd] = read_time_signal_csv(bgnd_file);
        % Check if background time vector is consistent
        if ~isequal(time, time_bgnd)
            error('Time vector in background file is not consistent with signal files.');
        end
    else
        bgnd = [];  % No background provided
    end

    figure;
    hold on;

    % Loop through each signal
    for k = 1:nSignals
        % If background is provided, clean the signal
        if ~isempty(bgnd)
            clear_signal = cleaning_signal(time, signals{k}, bgnd);
        else
            % If no background, use the original signal
            clear_signal = signals{k};
        end
        
        % Compute FFT (assuming computeFFT returns freq in THz)
        [freq, fft_data] = computeFFT(time, clear_signal);
        % Plot the magnitude spectrum
        plot(freq, abs(fft_data), 'LineWidth', 1.5);
    end

    hold off;
    xlabel('Frequency (THz)');
    ylabel('Magnitude');
    title('FFT Spectrum of Clean Signals');
    grid on;
    % Limit x-axis to half of the maximum frequency for clarity
    xlim([min_freq, max_freq]);
    % Add legend with provided labels
    legend(labels, 'Location', 'Best');
end


function clear_signal = cleaning_signal(time, signal, bgnd, plotFlag)
% cleaning_signal - Subtracts the background signal from the original signal.
%
% Syntax: 
%   clear_signal = cleaning_signal(time, signal, bgnd)
%   clear_signal = cleaning_signal(time, signal, bgnd, plotFlag)
%
% Inputs:
%    time     - A vector of time values
%    signal   - The original signal vector
%    bgnd      - The background signal vector to subtract from the original signal
%    plotFlag - (Optional) Boolean flag to plot the result (true to plot,
%               false to disable plotting). Default is false.
%
% Output:
%    clear_signal - The resulting signal after subtraction (signal - bgnd)
%
% Example:
%    t = 0:0.001:1-0.001;          % Time vector
%    signal = sin(2*pi*50*t);       % Original 50 Hz sine wave
%    bgnd = 0.5*sin(2*pi*50*t);      % Background signal (scaled 50 Hz sine wave)
%    % Without plotting:
%    clear_signal = cleaning_signal(t, signal, bgnd);
%    % With plotting:
%    clear_signal = cleaning_signal(t, signal, bgnd, true);

    % Set default plotFlag to false if not provided
    if nargin < 4
        plotFlag = false;
    end

    % Ensure that inputs are column vectors for consistency
    time = time(:);
    signal = signal(:);
    bgnd = bgnd(:);

    % Check if all vectors are of the same length
    if length(time) ~= length(signal) || length(time) ~= length(bgnd)
        error('Time, signal, and bgnd vectors must be of the same length.');
    end

    % Subtract the background from the signal
    clear_signal = signal - bgnd;

    % Plot the resulting clear signal only if plotFlag is true
    if plotFlag
        figure;
        plot(time, clear_signal, 'LineWidth', 1.5);
        xlabel('Time (s)');
        ylabel('Amplitude');
        title('Clear Signal (Signal - Background)');
        grid on;
    end
end

function [freq, fft_data] = computeFFT(time, signal)
% computeFFT - Computes the FFT of a time-domain signal and normalizes
%             the output with respect to the time increment.
%
% Syntax:  [freq, fft_data] = computeFFT(time, signal)
%
% Inputs:
%    time   - A vector of time values (assumed to be uniformly spaced)
%    signal - A vector containing the signal values corresponding to the time vector
%
% Outputs:
%    freq     - Frequency vector corresponding to the FFT bins (in THz)
%    fft_data - FFT of the input signal, normalized with respect to dt
%
% Example:
%    t = 0:0.001:1-0.001;         % Time vector (1 second at 1 kHz sampling rate)
%    x = sin(50*t);               % 50 Hz sine wave
%    [f, X] = computeFFT(t, x);
%    plot(f, abs(X));
%    xlabel('Frequency (THz)');
%    ylabel('Magnitude');
%    title('FFT of the Signal');

    % Ensure the input vectors are columns for consistency
    time = time(:);
    signal = signal(:);

    % Calculate the sampling interval (assuming uniform sampling)
    dt = time(2) - time(1);
    
    % Number of samples in the signal
    n = length(signal);
    
    % Compute the FFT of the signal and normalize with dt
    fft_data = fft(signal) * dt;
    
    % Create the frequency vector.
    % Frequency resolution is 1/(n*dt), converted to THz by multiplying by 2*pi.
    freq = (0:n-1)' * (1/(n*dt)) * 2 * pi;
    
    % Optionally, if you want to shift the zero-frequency component to the center,
    % uncomment the following lines:
    % fft_data = fftshift(fft_data);
    % freq = (-n/2:n/2-1)' * (1/(n*dt)) * 2 * pi;
end

function plotTimeDomain(filename)
% plotTimeDomain - Plots the time-domain signal.
%
% Syntax: plotTimeDomain(time, signal)
%
% Inputs:
%    filename - The name of the CSV file to read (including extension)
%
    [time, signal] = read_time_signal_csv(filename)
    figure;
    plot(time, signal, 'LineWidth', 1.5);
    xlabel('Time (ps)');
    ylabel('Amplitude');
    title('Time Domain Signal');
    grid on;
end

function [time, signal] = read_time_signal_csv(filename)
% read_time_signal_csv - Reads a CSV file containing time and signal data.
% The first column is assumed to be time, and the second column is the signal.
% The first row is ignored (assumed to be a header).
%
% Syntax: [time, signal] = read_time_signal_csv(filename)
%
% Inputs:
%    filename - The name of the CSV file to read (including extension)
%
% Outputs:
%    time    - Vector containing time values
%    signal  - Vector containing corresponding signal values
%
% Example:
%    [time, signal] = read_time_signal_csv('data.csv');
%    plot(time, signal);
%    xlabel('Time (s)');
%    ylabel('Signal');
%    title('Time-Domain Signal');
%
% Assumption:
%    - The CSV file has at least two columns with numerical data.
%    - The first row is a header and will be ignored.

    % Check if the file exists
    if ~isfile(filename)
        error('File not found: %s', filename);
    end
    
    % Read the data, ignoring the first row
    data = readmatrix(filename, 'NumHeaderLines', 1);
    
    % Check if the data has at least two columns
    if size(data, 2) < 2
        error('The CSV file must have at least two columns.');
    end
    
    % Extract time and signal columns
    time = data(:, 1);
    signal = data(:, 2);
end
