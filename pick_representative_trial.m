function idx = pick_representative_trial(pos_err)
% pick_representative_trial  Select the trial whose RMSE is closest to median.
%   pos_err : (n_mc, N) array, NaN for missing steps
rmse_per_trial = sqrt(nanmean(pos_err.^2, 2));   % (n_mc,)
med_val = median(rmse_per_trial);
[~, idx] = min(abs(rmse_per_trial - med_val));
end
