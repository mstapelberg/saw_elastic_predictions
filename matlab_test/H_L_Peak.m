function [y]=H_L_Peak(var,psaw)
%This function is to return the RW peak location

%y: RW peak slowness position
%var: displacement profile

%Find SAW by finding positions in displacement vs slowness that have a
%negative instantaneous slope
PeakTotal=find((diff(var))<0);
if isempty(PeakTotal)
    y=0;
elseif psaw
    y=[PeakTotal(end) PeakTotal(end-1)];
else
y=(PeakTotal(end));
end

end

