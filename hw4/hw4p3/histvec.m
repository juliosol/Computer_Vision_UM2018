function v = histvec(image,mask,b)
%
%  W18 EECS 504 HW4p2 Fg-bg Graph-cut
%  Jason Corso, jjcorso@umich.edu
%
%  For each channel in the image, compute a b-bin histogram (uniformly space
%  bins in the range 0:1) of the pixels in image where the mask is true. 
%  Then, concatenate the vectors together into one column vector (first
%  channel at top).
%
%  mask is a matrix of booleans the same size as image.
% 
%  normalize the histogram of each channel so that it sums to 1.
%
%  You CAN use the hist function.
%  You MAY loop over the channels.

chan = size(image,3);
vert_pixels = size(image,1);
hor_pixels = size(image,2);

c = 1/b;       % bin offset
x = c/2:c:1;   % bin centers

%%% FILL IN THE CODE HERE
% The output v should be a 3*b vector because we have a color image and 
% you have a separate histogram per color channel.

pre_vector = [];
for i = 1 : chan
    current_im_chan = image(:,:,i);
    vector_pixels_taken = [];
    for y = 1 : vert_pixels
        for z = 1 : hor_pixels
            if mask(y,z) == 1
                vector_pixels_taken = [vector_pixels_taken; [y,z]];
            end
        end
    end
    num_pixels = length(vector_pixels_taken);
    vector_for_hist = [];
    for k = 1 : num_pixels
        vector_for_hist = [vector_for_hist; current_im_chan(vector_pixels_taken(k,1),vector_pixels_taken(k,2))];
    end
    histogram_vector = hist(vector_for_hist,x);
    normalized_vector = histogram_vector/norm(histogram_vector);
    %bar(normalized_vector);
    pre_vector = [pre_vector, normalized_vector];
%%%

end
v = pre_vector';
%bar(v)
end