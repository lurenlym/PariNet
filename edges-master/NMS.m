function pickLocate = NMS(boxes, overlap)
 
% Non-maximum suppression.
% In object detect algorithm, select high score detections and skip windows
% covered by a previously selected detection.
%
% input - boxes : object detect windows.
%                 xMin yMin xMax yMax score.
%         overlap : suppression threshold.
% output - pickLocate : number of local maximum score.
 
boxes = double(boxes);
 
if isempty(boxes)
    pickLocate = [];
else
    xMin = boxes(:, 1);
    yMin = boxes(:, 2);
    xMax = boxes(:, 3)+boxes(:, 1);
    yMax = boxes(:, 4)+boxes(:, 2);
     
    s = boxes(:, end);
     
    % area of every detected windows.
    area = (xMax - xMin + 1) .* (yMax - yMin + 1);
     
    % sort detected windows based on the score.
    [vals, I] = sort(s);
     
    pickLocate = [];
    while ~isempty(I)
        last = length(I);
        i = I(last);
         
        pickLocate = [pickLocate; i];
        suppress = [last];
        if i == 114
           temp=0;
        end 
        for pos = 1 : last - 1
            
            j = I(pos); 
            
            % covered area.
            xx1 = max(xMin(i), xMin(j));
            yy1 = max(yMin(i), yMin(j));
            xx2 = min(xMax(i), xMax(j));
            yy2 = min(yMax(i), yMax(j));
             
            w = xx2 - xx1 + 1;
            h = yy2 - yy1 + 1;
             
            if ((w > 0) && (h > 0))
                % compute overlap.
                o = w * h / min(area(i), area(j));
                 
                if (o > overlap)
                    if j == 114
                        temp=0;
                    end 
                    suppress = [suppress; pos];
                end
            end
             
            % xx1 = max(x1(i), x1(I(1:last-1)));
            % yy1 = max(y1(i), y1(I(1:last-1)));
            % xx2 = min(x2(i), x2(I(1:last-1)));
            % yy2 = min(y2(i), y2(I(1:last-1)));
             
            % w = max(0.0, xx2-xx1+1);
            % h = max(0.0, yy2-yy1+1);
             
            % inter = w.*h;
            % o = inter ./ (area(i) + area(I(1:last-1)) - inter);
             
            % saving the windows which o less than threshold.
            % I = I(o <= overlap);
        end
        if  I(suppress)  == 114
                temp=0;
         end 
        I(suppress) = [];
    end
end