function [B] = graphcut(segmentimage,segments,keyindex)
%
% W18 EECS 504 HW4p2 Fg-bg Graph-cut
% Jason Corso, jjcorso@umich.edu
%
% Function to take a superpixel set and a keyindex and convert to a 
%  foreground/background segmentation.
%
% keyindex is the index to the superpixel we wish to use as foreground and
% find its relevant neighbors that would be in the same macro-segment
%
% Similarity is computed based on segments(i).fv (which is a color histogram)
%  and spatial proximity.
%
% segmentimage and segments are returned by the superpixel function
%  segmentimage is called S and segments is called Copt  
%
% OUTPUT:  B is a binary image with 1's for those pixels connected to the
%          source node and hence in the same segment as the keyindex.
%          B has 0's for those nodes connected to the sink.

%% Compute basic adjacency information of superpixels
%%%  Note that segNeighbors is code you need to implement
adjacency = j_segNeighbors(segmentimage);
%adjacency = segNeighbors(segmentimage);
%debug
%figure; imagesc(adjacency); title('adjacency');

% Normalization for distance calculation based on the image size.
% For points (x1,y1) and (x2,y2), distance is exp(-||(x1,y1)-(x2,y2)||^2/dnorm)
% Thinking of this like a Gaussian and considering the Std-Dev of the Gaussian 
% to be roughly half of the total number of pixels in the image. Just a
% guess.
dnorm = 2*prod(size(segmentimage)/2)^2;

k = length(segments);

%% Generate capacity matrix
capacity = zeros(k+2,k+2);  % initialize the zero-valued capacity matrix
source = k+1;  % set the index of the source node
sink = k+2;    % set the index of the sink node

% This is a single planar graph with an extra source and sink.
%
% Capacity of a present edge in the graph (adjacency) is to be defined as the product of
%  1:  the histogram similarity between the two color histogram feature vectors.
%      use the provided histinterset function below to compute this similarity 
%  2:  the spatial proximity between the two superpixels connected by the edge.
%      use exp(-D(a,b)/dnorm) where D is the euclidean distance between superpixels a and b,
%      dnorm is given above.
%
% * Source gets connected to every node except sink:
%   capacity is with respect to the keyindex superpixel
% * Sink gets connected to every node except source:
%   capacity is opposite that of the corresponding source-connection (from each superpixel)
%   in our case, the max capacity on an edge is 3; so, 3 minus corresponding capacity
% * Other superpixels get connected to each other based on computed adjacency
% matrix:
%  capacity defined as above. EXCEPT THAT YOU ALSO NEED TO MULTIPLY BY A SCALAR 0.25 for
%  adjacent superpixels.


%%% FILL IN CODE HERE to generate the capacity matrix using the description above.

number_regions = k;

for a = 1 : k
    for b = 1:k
        if  adjacency(a,b) == 1
            hist_similarity = histintersect(segments(a).fv,segments(b).fv);
            point_1 = [segments(a).x;segments(a).y];
            point_2 = [segments(b).x; segments(b).y];
            spatial_proximity = exp(-(norm(point_1-point_2)^2)/dnorm);
            capacity_value = 0.25*hist_similarity*spatial_proximity;
            capacity(a,b) = capacity_value;
        end 
    end
end

%capacity
%size(capacity)

% Computing source capacity to every other possible vertex.

capacity_source = zeros(k,1);
for a = 1:k
    hist_similarity = histintersect(segments(a).fv,segments(keyindex).fv);
    point_1 = [segments(a).x;segments(a).y];
    point_2 = [segments(keyindex).x;segments(keyindex).y];
    spatial_proximity = exp(-(norm(point_1-point_2)^2)/dnorm);
    capacity_value = hist_similarity * spatial_proximity;
    capacity_source(a) = capacity_value;
end

capacity_sink = 3*ones(k,1) - capacity_source;

capacity(1:k,k+2) = capacity_sink;
capacity(k+1,1:k) = capacity_source';

%%%

%debug
%figure; imagesc(capacity); title('capacity');
%y = 1;


%% Compute the cut (this code is provided to you)
[~,current_flow] = ff_max_flow(source,sink,capacity,k+2);

%% Extract the two-class segmentation.
% The cut will separate all nodes into those connected to the
%  source and those connected to the sink.
% The current_flow matrix contains the necessary information about
%  the max-flow through the graph.
% 
% Populate the binary matrix B with 1's for those nodes that are connected
%  to the source (and hence are in the same big segment as our keyindex) in the
%  residual graph.
% 
% You need to compute the set of reachable nodes from the source.  Recall, from
%  lecture that these are any nodes that can be reached from any path from the
%  source in the graph with residual capacity (original capacity - current flow) 
%  being positive.

%%%  FILL IN CODE HERE to read the cut into B

size_image = size(segmentimage);
ver_size = size_image(1);
hor_size = size_image(2);
residual_graph = capacity - current_flow;

B = zeros(ver_size,hor_size);

s = java.util.Stack();
s.push(k+1);
connected_source = [];
discovered_spixels = [];
while s.empty() ~= 1
    v = s.pop();
    connection_vector = residual_graph(v,1:k);
    if ismember(v,discovered_spixels) ~= 1
        discovered_spixels = [discovered_spixels;v];
        for i = 1 : length(connection_vector)
            if residual_graph(v,i) > 0
                connected_source = [connected_source; i];
                s.push(i);
            end
        end
    end
end

connection_source = unique(connected_source);
    
%end

for x = 1 : ver_size
    for y = 1:hor_size
        coordinate_considered = segmentimage(x,y);
        for i = 1 : length(connection_source)
            if coordinate_considered == connection_source(i)
                B(x,y) = 1;
            end
        end        
    end
end

%%%

end


function c = histintersect(a,b)
    c = sum(min(a,b));
end
