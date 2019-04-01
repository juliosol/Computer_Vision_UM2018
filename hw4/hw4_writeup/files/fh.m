function seg = fh(G, k, seg_img)
%
% W18 EECS 504 HW4p3 Image Segmentation with Minimum Spanning Forest
%
% Felzenszwalb-Huttenlocher algorithm implementation, which is a modified
% version of the Kruskal algorithm.
% Input  G: mx3 matrix, the adjacency list of a graph, of which each row is 
%           an edge (node1, node2, weights). m is the number of edges.
%        k: hyperparameter for F-H algorithm
% Output seg: 1xn vector, the segment id assigned to each node in the graph
%             n is the number of nodes, id is from 1 to number of segments.
%

sorted_matrix = sortrows(G,3);
size_matrix = size(sorted_matrix);
no_edges = size_matrix(1);

pre_edges = [G(:,1);G(:,2)];
edges = unique(pre_edges);
no_vertices = length(edges);

% Creation of components 
components_vector = {};
for t = 1:no_vertices
    components_vector{t} = [edges(t), 0];
end
%components_vector

% In this part we will construct the segments
p = 1
for q = 1 : no_edges
    current_edge = sorted_matrix(q,:);
    current_v1 = current_edge(1);
    current_v2 = current_edge(2);
    current_weight = current_edge(3);

    for s = 1:no_vertices
        %f p == 150
         %  a = 3; 
        %end
        %p = p + 1
        if ismember(current_v1,components_vector{s}) == 1
            component_1_index = s;
        end
        if ismember(current_v2,components_vector{s}) == 1
            component_2_index = s;
        end
    end
    
    component_1 = components_vector{component_1_index};
    component_2 = components_vector{component_2_index};
    
    %internal_comp1 = component_1(end);
    %internal_comp2 = component_1(end);
    %tau_comp1 = k/(length(component_1) - 1);
    %tau_comp2 = k/(length(component_2) - 1);
    
    %minimum_internal = min(internal_comp1 + tau_comp1,internal_comp2 + tau_comp2);
    
    if isequal(component_1(1:end-1),component_2(1:end-1)) == 0 
        
        internal_comp1 = component_1(end);
        internal_comp2 = component_2(end);
        tau_comp1 = k/(length(component_1) - 1);
        tau_comp2 = k/(length(component_2) - 1);
    
        minimum_internal = min(internal_comp1 + tau_comp1,internal_comp2 + tau_comp2);
        
        if current_weight <= minimum_internal
            nC = {};
            n = 1;
            for j = 1 : length(components_vector)
                if j ~= component_1_index && j ~= component_2_index
                    nC{n} = components_vector{j};
                    n = n + 1;
                end
            end
            
            max_weight_edge = max([internal_comp1,internal_comp2,current_weight]);
            nC{n} = [component_1(1:end-1),component_2(1:end-1),max_weight_edge];
            
            %nC{n} 
            %components_vector{component_1_index} = [component_1(1:end-1), component_2(1:end-1),max_weight_edge];
            %components_vector(component_2_index) = [];
            %no_vertices = no_vertices - 1;
        end
    end
    components_vector = nC;
    size(components_vector)
    %if size(components_vector,2) <= 52
    %    a = 2
    %end
    no_vertices = size(components_vector,2);
end

no_vertices = length(edges);
%components_vector;
size(components_vector);
no_components = length(components_vector);
zsegment_vertex_vector = zeros(1,no_vertices);

for j = 1 : no_components
    current_component = components_vector{j};
    %j
    for p = 1 : length(current_component(1:end-1))
       %p
       segment_vertex_vector(current_component(p)) = j;
    end
end

%To run the algorithm with the post_processing, just uncomment the line
%calling the post_processing function and rename the first seg in line 88
%to be pre_seg.

seg = segment_vertex_vector;
%c = 1;
%seg = post_processing(sorted_matrix,pre_seg,seg_img,3);
%seg;
%w = 1;
end

%%% This part is the implementation of the post_processing function.

function seg_final = post_processing(sorted_matrix,seg,seg_img,min_size)

% IN this function we take the sorted matrix, the result from the function
% above and the whole image with the segment on each pixel. 
   size_matrix = size(sorted_matrix);
   no_edges = size_matrix(1);
   size_segments = size(seg);
   for j = no_edges : -1 : 1
       edge_considered = sorted_matrix(j,:);
       vertex1 = edge_considered(1);
       vertex2 = edge_considered(2);
       component_vertex_1 = seg(vertex1);
       vector_spixels_1 = find(seg(:) == component_vertex_1);
       component_vertex_2 = seg(vertex2);
       vector_spixels_2 = find(seg(:) == component_vertex_2);
       size_comp_1 = length(vector_spixels_1);
       size_comp_2 = length(vector_spixels_2);
       
       if component_vertex_1 ~= component_vertex_2
          if size_comp_1 < min_size
              for x = 1 : size_comp_1
                  seg(1,vector_spixels_1(x)) = component_vertex_1;
              end
          elseif size_comp_2 < min_size 
              for x = 1 : size_comp_2
                  seg(1,vector_spixels_2(x)) = component_vertex_1;
              end
          end
       end
   end
   seg_final = seg;
end
