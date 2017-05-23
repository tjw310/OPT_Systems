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
    
    %% private methods
    methods (Access = private)
        %constructs vector from obj.x, obj.z coordinates
        function out = vectorXZ(obj)
            out = [obj.x;obj.z];
        end
    end
    
    %% static methods
    methods (Static)
        % calculates rotation matrix (2D) by angle theta
        %@ param double theta, angle of rotation in radians
        function R = getRMatrix(theta)
            R = [cos(theta),-sin(theta);sin(theta),cos(theta)];
        end
        
        % returns inverse of 2D matrix
        %@ param double[][] matrix, 2x2 array of doubles
        function out = inverse(matrix)
            det = matrix(2,2)*matrix(1,1)-matrix(1,2)*matrix(2,1);
            out = 1/det.*[matrix(2,2),-matrix(1,2);-matrix(2,1),matrix(1,1)];
        end
        
        %@param double theta, angle to rotate
        %@param double[] inputVector, vector to translate
        function outputVector = rotate(theta,inputVector)
            R = PointObject.getRMatrix(theta);
            outputVector = R*inputVector;
        end
        
        %@param double d0,d1 translation amounts in first and second
        %dimensions
        %@param double[] inputVector, vector to translate
        function outputVector = translate(d0,d1,inputVector)
            outputVector = inputVector-[d0;d1];
        end
            
    end
    %% point manipulation methods
    methods
        % @param double projectionAngle, projection angle (+ anti-clock rotation, -
        % clockwise roation)
        % @param double[] varargin, [centreOfRotationX,centreOfRotationZ] if displaced rotation
        % axis in mm
        function [xRot,zRot] = getRotatedXZ(obj,projectionAngle,varargin)
            if nargin == 3
                if isnumeric(varargin{1})
                     centre = varargin{1};
                     centreOfRotationX = centre(1);
                     centreOfRotationZ = centre(2);
                     rotCoords = PointObject.translate(-centreOfRotationX,-centreOfRotationZ)*PointObject.rotate(projectionAngle)*PointObject.translate(centreOfRotationX,centreOfRotationZ)*obj.vectorXZ;
                     xRot = rotCoords(1);
                     zRot = rotCoords(2);
                else
                    error('Enter x centre of rotation locations or leave blank');
                end
            else
                rotCoords = PointObject.rotate(projectionAngle)*obj.vectorXZ;
                 xRot = rotCoords(1);
                 zRot = rotCoords(2);
            end
        end      
    end
end

