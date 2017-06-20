classdef StepperMotor < handle
    %Defines stepper motor properties such as rotation axis location,
    %angle in xy plane, gyroscopic motion over the OPT acquisition cycle
    
    properties (Access = private)
        x % x displacement from centre of object space volume in mm
        z % z displacement from centre of object space volume in mm
        angle % angle of rotation axis in the xy plane in radians
        xMotion % x displacement from average position over projection cycle in mm, [] if no motion
        zMotion % z displacement from average position over projection cycle in mm, [] if no motion
    end
    
    %% constructor and get/set
    methods
        function obj = StepperMotor(varargin)
            if nargin==5
                obj.x = varargin{1};
                obj.z = varargin{2};
                obj.angle = varargin{3};
                obj.xMotion = varargin{4};
                obj.zMotion = varargin{5};
            elseif nargin==0
                obj.x = 0;
                obj.z = 0;
                obj.angle = 0;
                obj.xMotion = [];
                obj.zMotion = [];
            else
                error('Enter 0 or 5 arguments');
            end
        end
        
        %@param double x
        function setX(obj,x)
            obj.x = x;
        end
        %@param double z
        function setZ(obj,z)
            obj.z=z;
        end
        %@param double angle
        function setAngle(obj,angle)
            obj.angle = angle;
        end
        
        %@param double[] xMotion, 1D array of length nProj
        %@param double nProj, number of projections
        function setXMotion(obj,xMotion,nProj)
            if length(xMotion)~=nProj
                error('Length of motion array must equal number of projections');
            else
                obj.xMotion = xMotion;
                if isempty(obj.zMotion)
                    obj.zMotion = zeros(size(xMotion));
                end
            end
        end
        
        %@param double[] zMotion, 1D array of length nProj
        %@param double nProj, number of projections
        function setZMotion(obj,zMotion,nProj)
            if length(zMotion)~=nProj
                error('Length of motion array must equal number of projections');
            else
                obj.zMotion = zMotion;
                if isempty(obj.xMotion)
                    obj.xMotion = zeros(size(zMotion));
                end
            end
        end
        
        function clearMotion(obj)
            obj.xMotion = [];
            obj.zMotion = [];
        end
        
        %@return double
        function out = getX(obj)
            out = obj.x;
        end
        %@return double
        function out = getZ(obj)
            out = obj.z;
        end
        %@return double
        function out = getAngle(obj)
            out = obj.angle;
        end
        %@return double[]
        function out = getXMotion(obj)
            out = obj.xMotion;
        end
        %@return double[]
        function out = getZMotion(obj)
            out = obj.zMotion;
        end
        %@param double nProj, total number of projections
        %@return double[]
        function out = getTotalX(obj,nProj)
            if ~isempty(obj.xMotion)
                out = obj.xMotion+obj.x;
            else
                out = repmat(obj.x,1,nProj);
            end
        end
        %@param double nProj, total number of projections
        %@return double[]
        function out = getTotalZ(obj,nProj)
            if ~isempty(obj.zMotion)
                out = obj.zMotion+obj.z;
            else
                out = repmat(obj.z,1,nProj);
            end
        end
        
        function plotMotion(obj)
            figure; subplot(1,2,1); plot(obj.xMotion);
            subplot(1,2,2); plot(obj.zMotion);
        end
    end
    
    %% 
    methods
        %@param double projectionNumber
        %@return double
        function out = currentX(obj,projectionNumber)
            if ~isempty(obj.xMotion)
                out = obj.x+obj.xMotion(projectionNumber);
            else
                out = obj.x;
            end
        end
        %@param double projectionNumber
        %@return double
        function out = currentZ(obj,projectionNumber)
            if ~isempty(obj.zMotion)
                out = obj.z+obj.zMotion(projectionNumber);
            else
                out = obj.z;
            end
        end
 
        % rotates a pointobject using motor axis properties
        %@param PointObject object, point object to rotate
        %@param double projectionNumber, projectionNumber
        %@param double[] theta, array of projection angles
        %@return PointObject rotatedObject, returns point object
        function [rotatedObject,x,y,z] = rotate(obj,object,projectionNumber,theta)
             Rxy = PointObject.Rxy(obj.angle);
             T = PointObject.T(obj.currentX(projectionNumber),0,obj.currentZ(projectionNumber));
             Rxz = PointObject.Rxz(theta(projectionNumber));
             rotCoords = (inv(Rxy))*T*Rxz*(inv(T))*Rxy*object.vectorXYZ;
             rotatedObject = PointObject(rotCoords(1),rotCoords(2),rotCoords(3));
             x = rotCoords(1); y = rotCoords(2); z = rotCoords(3);
        end
        
        % rotates a pointobject using motor axis properties, FIRST rotates
        % object into xy-plane and then rotates around z axis. i.e same
        % method as above but does not rotate back.
        %@param PointObject object, point object to rotate
        %@param double projectionNumber, projectionNumber
        %@param double[] theta, array of projection angles
        %@return PointObject rotatedObject, returns point object
        function [rotatedObject,x,y,z] = rotate0(obj,object,projectionNumber,theta)
             Rxy = PointObject.Rxy(obj.angle);
             T = PointObject.T(obj.currentX(projectionNumber),0,obj.currentZ(projectionNumber));
             Rxz = PointObject.Rxz(theta(projectionNumber));
             rotCoords = T*Rxz*(inv(T))*Rxy*object.vectorXYZ;
             rotatedObject = PointObject(rotCoords(1),rotCoords(2),rotCoords(3));
             x = rotCoords(1); y = rotCoords(2); z = rotCoords(3);
        end
        
        % calculates x displacement from ideal sinusoid trace (assuming
        % that obj.angle==0
        %@param double[] theta, array of projection angles
        function out = xDisplacement(obj,theta)
            if obj.angle==0
                if isempty(obj.xMotion) || isempty(obj.zMotion)
                    out = 0;
                else
                    out = obj.xMotion.*(1-cos(theta))+obj.zMotion.*sin(theta);
                end
            else
                error('AoR angle must be zero');
            end
        end
        
        % calculates z displacement from ideal sinusoid trace (assuming
        % that obj.angle==0
        %@param double[] theta, array of projection angles
        function out = zDisplacement(obj,theta)
            if obj.angle==0
                if isempty(obj.xMotion) || isempty(obj.zMotion)
                    out = 0;
                else
                    out = obj.zMotion.*(1-cos(theta))-obj.xMotion.*sin(theta);
                end
            else
                error('AoR angle must be zero');
            end
        end
    end
end

