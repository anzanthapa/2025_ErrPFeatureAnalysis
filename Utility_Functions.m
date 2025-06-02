classdef Utility_Functions
    %UTILITY_FUNCTIONS Summary of this class goes here
    %   This class will have the functions used for ErrP analysis.
    methods(Static)


        function [HPF_signal,BPF_signal] = preprocess_eeg(raw_eeg,fs)
            HPF_signal = zeros(size(raw_eeg));
            BPF_signal = zeros(size(raw_eeg));
            for chi = 1:size(raw_eeg,2)
                [HPF_signal(:,chi),~] = highpass(raw_eeg(:,chi)',0.1,fs,"ImpulseResponse","iir",'StopbandAttenuation',60);
                [BPF_signal(:,chi),~] = bandpass(raw_eeg(:,chi)',[1,10],fs,"ImpulseResponse","iir",'StopbandAttenuation',60);
            end
        end


        function [dwt_feature]=calculate_WPD(inputEEG)
            waveletname='db4';
            decomplevel=3;
            dwt_feature_array=[];
            for chi=1:size(inputEEG,1)
                wpd_trees=wpdec(inputEEG(chi,:),decomplevel,waveletname);
                dwt_feature_array1=wpcoef(wpd_trees,[3,0]);
                dwt_feature_array2=wpcoef(wpd_trees,[3,1]);
                dwt_feature_array=[dwt_feature_array;dwt_feature_array1,dwt_feature_array2];
            end
            dwt_feature=reshape(dwt_feature_array',1,[]);
        end

        function psd_feature=calculate_PSD(inputEEG,fs)
            L=size(inputEEG,2);
            psd_feature_array=zeros(size(inputEEG,1),L/2+1);
            for chi=1:size(inputEEG,1)
                fftsig=fft(inputEEG(chi,:));
                psd=(1/(fs*L))*abs(fftsig).^2;
                psd_onesided = psd(1:L/2+1);
                psd_onesided(2:end-1)=2*psd_onesided(2:end-1);
                psd_feature_array(chi,:)=psd_onesided;
            end
            psd_feature = reshape(psd_feature_array',1,[]);
        end
        %% Spectral Entropy
        function [spectEn] = func_compute_SpectEn(inputEEG,fs)

            % Compute Spectral Entropy for each row (channel) of the EEG
            % Segment
            % inputEEG: Input EEG signal (size: numChannels x EEG samples)
            % spectEn: Output spectral entropy (size: 1 x numChannels)
            L=size(inputEEG,2);
            PSD=zeros(size(inputEEG,1),L/2+1);
            for chi=1:size(inputEEG,1)
                fftsig=fft(inputEEG(chi,:));
                psd=(1/(fs*L))*abs(fftsig).^2;
                psd_onesided = psd(1:L/2+1);
                psd_onesided(2:end-1)=2*psd_onesided(2:end-1);
                PSD(chi,:)=psd_onesided;
            end
            % Ensure PSD values are positive to avoid log issues
            PSD(PSD == 0) = eps; % Replace zero values with machine epsilon
            % Compute total power per row (sum over frequency bins)
            totalPSD = sum(PSD, 2); % Sum along columns (across frequencies)

            % Compute normalized power (probability distribution)
            PWi = PSD./ totalPSD;

            % Compute Spectral Entropy
            spectEn = -sum(PWi .* log(PWi), 2); % Sum across frequency dimension using natural log (ln)
        end

        function entropy = calculate_shannon_entropy(signal, n_bins)
            % Set default number of bins if not specified
            if nargin < 2
                n_bins = Utility_Functions.calculate_sturges_rule(size(signal, 1));
            end

            % Calculate histogram of the signal
            % histcounts returns the counts and edges of the bins
            [counts, ~] = histcounts(signal, n_bins);

            % Convert counts to probabilities by dividing by total number of samples
            probabilities = counts / sum(counts);

            % Remove zero probabilities to avoid log(0)
            probabilities = probabilities(probabilities > 0);

            % Calculate Shannon entropy: -sum(p * log2(p))
            entropy = -sum(probabilities .* log2(probabilities));
        end

        function entropy_feature = calculate_Entropy(inputEEG, fs)
            [b, a] = butter(4, [4 8]./(fs/2), 'bandpass');  % Design filter once
            num_channels = size(inputEEG, 1);
            entropy_feature_array = zeros(1,num_channels);  % Preallocate
            for chi = 1:num_channels
                % Filter to extract theta band
                theta_band = filtfilt(b, a, inputEEG(chi, :));

                % Use default number of bins if not specified
                n_bins = round(1 + log2(length(theta_band)));
                % Histogram to estimate probabilities
                [counts, ~] = histcounts(theta_band, n_bins);
                probabilities = counts./ sum(counts);

                % Remove zero entries
                probabilities = probabilities(probabilities > 0);

                % Shannon entropy
                entropy_feature_array(1,chi) = -sum(probabilities .* log2(probabilities));
            end

            entropy_feature = entropy_feature_array;
        end


        function aar_feature=calculate_AAR(inputEEG)
            P=12;
            num_channels=size(inputEEG, 1);
            for chi=1:num_channels
                aar_feature_array=ar(inputEEG(chi,:),P,'ls');
                aar_coeffs = aar_feature_array.A;
                aar_feature = [aar_feature, aar_coeffs];
            end
        end


        function [X_bal, Y_bal] = randomOversample(trainingX, trainingY)
            % randomOversample01  Oversample minority class (label=1) to match majority (label=0)
            %
            %   [X_bal, Y_bal] = randomOversample01(trainingX, trainingY)
            %
            %   Inputs:
            %     - trainingX : N×d feature matrix (rows = samples)
            %     - trainingY : N×1 label vector with values 0 (majority) or 1 (minority)
            %
            %   Outputs:
            %     - X_bal : M×d feature matrix after oversampling (M ≥ N)
            %     - Y_bal : M×1 label vector after oversampling
            %
            %   This duplicates (with replacement) random samples of class 1 until
            %   the number of class-1 samples equals the number of class-0 samples.
            %   If already balanced or no class-1 samples, returns inputs unchanged.

            % Ensure label vector is N×1
            if size(trainingY,2) ~= 1
                error('trainingY must be an N×1 vector.');
            end

            % Indices for majority (0) and minority (1)
            idx0 = find(trainingY == 0);
            idx1 = find(trainingY == 1);
            n0   = numel(idx0);  % majority count
            n1   = numel(idx1);  % minority count

            % If no minority or already balanced, return as is
            if n1 == 0 || n0 == n1
                X_bal = trainingX;
                Y_bal = trainingY;
                return;
            end

            % Number of additional class-1 samples needed
            delta = n0 - n1;  % positive since n0 > n1

            % Randomly sample (with replacement) from minority indices
            rnd_idx1 = randsample(idx1, delta, true);

            % Extract those minority samples
            X_minority_new = trainingX(rnd_idx1, :);   % delta×d
            Y_minority_new = trainingY(rnd_idx1);      % delta×1

            % Concatenate to original data
            X_bal = [trainingX; X_minority_new];   % (N + delta)×d
            Y_bal = [trainingY; Y_minority_new];   % (N + delta)×1

            % Shuffle rows so that classes are mixed
            M = size(X_bal, 1);
            perm = randperm(M);
            X_bal = X_bal(perm, :);
            Y_bal = Y_bal(perm);
        end



        function [accuracy, acc_class0, acc_class1, F1_class0, F1_class1] = binaryClassMetrics(predictions, trueLabels)
            % Ensure column vectors
            predictions = predictions(:);
            trueLabels = trueLabels(:);

            % Compute overall accuracy
            accuracy = sum(predictions == trueLabels) / numel(trueLabels);

            % Class 0
            TP0 = sum(predictions == 0 & trueLabels == 0);
            FP0 = sum(predictions == 0 & trueLabels == 1);
            FN0 = sum(predictions == 1 & trueLabels == 0);

            precision0 = TP0 / (TP0 + FP0 + eps);
            recall0 = TP0 / (TP0 + FN0 + eps);
            F1_class0 = 2 * (precision0 * recall0) / (precision0 + recall0 + eps);

            % Class 1
            TP1 = sum(predictions == 1 & trueLabels == 1);
            FP1 = sum(predictions == 1 & trueLabels == 0);
            FN1 = sum(predictions == 0 & trueLabels == 1);

            precision1 = TP1 / (TP1 + FP1 + eps);
            recall1 = TP1 / (TP1 + FN1 + eps);
            F1_class1 = 2 * (precision1 * recall1) / (precision1 + recall1 + eps);

            % Compute class specific accuracies
            N0 = sum(trueLabels==0);
            acc_class0 = TP0/(N0+eps);
            N1 = sum(trueLabels==1);
            acc_class1 = TP1/(N1+eps);
        end

    end
end
%% END OF SCRIPT

