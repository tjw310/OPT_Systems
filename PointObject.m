classdef PointObject
    % point object class, defines intial true object space location and
    % other
    
    properties (Access = private)
        x %object location (z is depth)
        y
        z
    end
    
    methods
        function obj = PointObject(x,y,z)
            obj.x = x; obj.y = y; obj.z = z;
        end
    end
    
    %% get/set methods
    methods 
        function out = getX(obj)
            out = obj.x;
        end
        function out = getY(obj)
            out = obj.y;
        end
        function out = getZ(obj)
            out = obj.z;
        end
        
        %@param SimpleConeRecon simpleConeRecon, class with system
        %information and recon information
        function out = getYinPixels(obj,simpleConeRecon,objective)
            out = obj.y/simpleConeRecon.getPixelSize*objective.getMagnification+simpleConeRecon.getHeight/2;
        end
        function out = getXinPixels(obj,simpleConeRecon,objective)
            out = obj.x/simpleConeRecon.getPixelSize*objective.getMagnification+simpleConeRecon.Width/2;
        end 
    end
    
    %% trace methods
    methods
        function out = getTrace(obj,simpleConeRecon,objective)
            theta = simpleConeRecon.theta;
            xRot = (obj.x*cos(theta)-obj.z*sin(theta))/simpleConeRecon.getPixelSize*objective.getMagnification;
            zRot = (obj.z*cos(theta)+obj.x*sin(theta))/simpleConeRecon.getPixelSize*objective.getMagnification;
            y = obj.y/simpleConeRecon.getPixelSize*objective.getMagnification;
            R = simpleConeRecon.getR;
            op = simpleConeRecon.getOpticCentre;
            
            epsilon = (xRot-op(1))./(1+zRot./R)+op(1)+simpleConeRecon.getWidth/2;
            sigma = (y-op(2))./(1+zRot./R)+op(2)+simpleConeRecon.getHeight/2;
            out(1,1:length(theta)) = epsilon;
            out(2,1:length(theta)) = sigma;
        end
            
    end
end

