function slope=limitedSlope(backwardGradient,forwardGradient,limiter)
%LIMITEDSLOPE Apply a named component-wise slope limiter.

if nargin<3 || isempty(limiter),limiter="minmod";end
limiter=lower(string(limiter));
switch limiter
    case "minmod"
        slope=tvd2.minmod(backwardGradient,forwardGradient);
    case {"mc","monotonized-central"}
        centered=.5*(backwardGradient+forwardGradient);
        doubled=tvd2.minmod(2*backwardGradient,2*forwardGradient);
        slope=tvd2.minmod(centered,doubled);
    case {"vanleer","van-leer"}
        slope=zeros(size(backwardGradient),'like', ...
            backwardGradient+forwardGradient);
        sameSign=(backwardGradient>0 & forwardGradient>0) | ...
            (backwardGradient<0 & forwardGradient<0);
        denominator=backwardGradient+forwardGradient;
        slope(sameSign)=2*backwardGradient(sameSign).* ...
            forwardGradient(sameSign)./denominator(sameSign);
    otherwise
        error('tvd2:Limiter','Unknown limiter: %s',limiter);
end
end

