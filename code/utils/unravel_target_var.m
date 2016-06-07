function ts = unravel_target_var(Y)

ts = reshape(flip(Y)', 1, numel(Y));
ts = ts';

end