function [smoothed score h]=smoothing(ydata,varargin)
%SMOOTHING smoothes noisy signal.
%   OUTPUT = SMOOTHING(INPUT) performs nonparametric kernel smoothing
%   of vector INPUT using Nadaraya-Watson, a local weighted regression.
%
%   OUTPUT = SMOOTHING(INPUT, H) smoothes INPUT with the given bandwidth H,
%   which controlls the smoothness of the estimate. If H is not specified
%   then H is optimized using leave-one-out cross-validation. 
%   The optimisation is performed with FMINSEARCH that returns 
%   a local optimum.
%
%   [OUTPUT, SCORE, H] = SMOOTHING(INPUT) smoothes INPUT and returns 
%   the estimate in OUTPUT vector. SCORE is the regression error from
%   leave-one-out cross-validation. H is the bandwidth used for regression.
%
%   Example
%   -------
%       load count.dat
%       data = count(:,1);
%       plot(data, 'r*')
%       hold on
%       plot(smoothing(data))
%
%   The example smoothes traffic counts at the first intersections
%   for each hour of the day.
%
%   See also SMOOTH, FILTER, FMINSEARCH, OPTIMSET.

%   Contributed by Jan Motl (jan@motl.us)
%   $Revision: 1.1 $  $Date: 2013/02/26 16:58:01 $


    % Check number of inputs.
    numvarargs = length(varargin);
    
    if numvarargs > 1
        error('smoothing:TooManyInputs', ...
            'requires at most 1 optional inputs: h');
    end
    
    % Check that the input is a vector.
    if ~isvector(ydata)
        error('The input must be a vector.');
    end
    
    % Check that the input is a column vector.
    if size(ydata,2)~=1
        ydata = ydata';
    end
    
    % Check that h is scalar.
    if numvarargs == 1 && ~isscalar(varargin{1})
        error('The h value must be scalar.');
    end
    
    % Either optimalise or use user defined value.
    if numvarargs == 1
        h = varargin{1};
        [score smoothed] = calculation(ydata, h);
    else
        options = optimset('Display','iter','TolX', 0.1);
        %h = fminbnd(@(h)calculation(ydata, h), 0.2, 100, options);
        h = fminsearch(@(h)calculation(ydata, h), 1, options);
        [score smoothed] = calculation(ydata,h);
        display(sprintf('The optimal bandwith H is: %f', h));
    end

    display(sprintf('The cross-validation score (smaller is better): %f', score)); 
end
    
  

function [score smoothed]=calculation(ydata,h)
    % Configuration.
    stdd=1;
    
    % Initialisation.
    len = length(ydata);    % length of the data
    xdata = 1:len;          % we assume eqi-distant samples to use tableK
    smoothed = ydata;       % to make sure the type class remains the same 
    
    % Look-up table (Gaussian value in a given distance).
    pdfAt = normpdf((0:len)/h, 0, stdd);

    % Another look-up table (reuse the denominator in cross-validation).
    denominators = zeros(len,1);
    for toSmooth = 1:ceil(len/2)    % exploit the symetry & calculate half
        denominators(toSmooth) = sum(pdfAt(abs(xdata(toSmooth)-xdata)+1));
    end
    denominators(len:-1:ceil(len/2)+1) = denominators(1:floor(len/2));
    
    % Calculation.
    for toSmooth = 1:len
        gammas = pdfAt(abs(xdata(toSmooth)-xdata)+1)/denominators(toSmooth);
        smoothed(toSmooth) = gammas*ydata;
    end
    
    % Leave-one-out cross-validation.    
    gammas = pdfAt(1)./denominators;
    score = sum(((ydata-smoothed)./(1-gammas)).^2);
    score = score/len;
end

