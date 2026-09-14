function value=minmod(left,right)
%MINMOD Overflow-safe component-wise minmod limiter.

if ~isequal(size(left),size(right))
    error('tvd2:LimiterSize','Limiter arguments must have equal size.');
end
if any(~isfinite(left),'all') || any(~isfinite(right),'all')
    error('tvd2:LimiterFinite','Limiter arguments must be finite.');
end
samePositive=left>0 & right>0;
sameNegative=left<0 & right<0;
sameSign=samePositive | sameNegative;
value=zeros(size(left),'like',left+right);
value(sameSign)=sign(left(sameSign)).* ...
    min(abs(left(sameSign)),abs(right(sameSign)));
end

