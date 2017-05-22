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
    end
    
    %% rotation methods
    methods
        % @param double projectionAngle, projection angle (+ anti-clock rotation, -
        % clockwise roation)
        % @param double varargin, centreOfRotationX if displaced rotation
        % axis
        function out = getRotatedX(obj,projectionAngle,varargin)
            centreOfRotationX = 0;
            if nargin > 2
                switch varargin
                    case isnumeric(varargin{1})
                        centreOfRotationX = varargin{1};
                    otherwise
                        error('Enter x centre of rotation locations or leave blank');
                end
            end
            out = (obj.x-centreOfRotationX)*cos(projectionAngle)-obj.z*sin(projectionAngle)+centreOfRotationX;
        end
        % @param double projectionAngle, projection angle (+ anti-clock rotation, -
        % clockwise roation)
        % @param double varargin, centreOfRotationZ if displaced rotation
        % axis
        function out = getRotatedZ(obj,projectionAngle,varargin)
            centreOfRotationZ = 0;
            if nargin > 2
                switch varargin
                    case isnumeric(varargin{1})
                        centreOfRotationZ = varargin{1};
                    otherwise
                        error('Enter x centre of rotation locations or leave blank');
                end
            end
            out = (obj.z-centreOfRotationZ)*cos(projectionAngle)+obj.x*sin(projectionAngle)+centreOfRotationZ;
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

