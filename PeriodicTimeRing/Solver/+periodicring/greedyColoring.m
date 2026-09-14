function [colors,colorCount]=greedyColoring(pattern)
%GREEDYCOLORING Color columns whose structural row sets do not overlap.

pattern=spones(sparse(pattern));
conflicts=spones(pattern.'*pattern);
conflicts=conflicts-spdiags(diag(conflicts),0, ...
    size(conflicts,1),size(conflicts,2));
degrees=full(sum(conflicts,2));
[~,order]=sort(degrees,'descend');
colors=zeros(size(pattern,2),1);
colorCount=0;
for orderIndex=1:numel(order)
    column=order(orderIndex);
    if nnz(pattern(:,column))==0
        continue
    end
    neighbours=logical(conflicts(:,column));
    unavailable=unique(colors(neighbours));
    unavailable(unavailable==0)=[];
    color=1;
    while any(unavailable==color),color=color+1;end
    colors(column)=color;
    colorCount=max(colorCount,color);
end
end
