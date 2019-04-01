function colorcircle(c, r, s, nsides)
%
% W18 EECS 504 HW3p1 Matching balloons with SIFT
% Draw a color circle.
%
% Arguments:  c -  A 2-vector [x y] specifying the centre.
%             r -  The radius.
%             s -  color of the line segments to draw [r g b] in [0 1]
%             nsides -  Optional number of sides in the polygonal approximation.
%                       (defualt is 16 sides)

if nargin == 2
    nsides = 16;
  s = [0 0 1];
elseif nargin == 3
  nsides = 16;
end

nsides = round(nsides);  % make sure it is an integer

a = [0:pi/nsides:2*pi];
h = line(r*cos(a)+c(1), r*sin(a)+c(2), 'LineWidth', 2);
set(h,'Color',s);
