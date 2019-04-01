function augmentpath=bfs_augmentpath(start,target,current_flow,capacity,n)
%
% W18 EECS 504 HW4p2 Fg-bg Graph-cut
% Find an augmentpath in a graph using breadth-first search
%

WHITE =0;
GRAY = 1;
BLACK = 2;
color(1:n) = WHITE;
q = [];
augmentpath = [];

% Enqueue
q = [start q];
color(start) = GRAY;

pred = zeros(1,n);
while ~isempty (q) 
    % Dequeue
    u = q(end);
    q(end) = [];
    color(u) = BLACK;

    for v=1:n
        if (color(v) == WHITE && capacity(u,v) > current_flow(u,v) )
            % Enqueue
            q = [v q];
            color(v) = GRAY;
            pred(v) = u;                        
        end
    end
end
if color(target) == BLACK % if target is accessible
    temp = target;
    while pred(temp) ~= start
        augmentpath = [pred(temp) augmentpath];  % augment path doesnt containt the start point AND target point
        temp=pred(temp);
    end
    augmentpath=[start augmentpath target];
else
    augmentpath=[];  % default resulte is empty
end
    
        


