classdef MagChange
    %class with properties relating the relative magnification between two
    %depth locations
    
    properties
        deltaMag %double, change in magnification
        error % double, error in change in mag
        z1 % double,  depth position 1 in mm
        z2 % double,  depth position 2 in mm
        path % string, path to folder containg calibration images
    end
    
    methods
        % CONSTRUCTOR
        function obj = MagChange(deltaMag,error,z1,z2,path)
            obj.deltaMag = deltaMag;
            obj.error = error;
            obj.z1 = z1;
            obj.z2 = z2;
            obj.path = path;
        end
    end
    
end



