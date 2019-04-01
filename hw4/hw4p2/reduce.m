function segments = reduce(image, segments, segmentimage)
%
% W18 EECS 504 HW4p2 Fg-bg Graph-cut
% Jason Corso, jjcorso@umich.edu
%
% Wrapper for function to compute feature descriptors on the segments
%

bins = 10;

for i=1:length(segments)

    segments(i).fv = j_histvec(image,segmentimage==i,bins);

end

