classdef PointObject
    % point object class, defines intial true object space location and
    % other
    % matrix transformations based on: http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
     
    properties (Access = private)
        x %object location (z is depth) in mm
        y
        z
    end
    
    methods
        function obj = PointObject(x,y,z)
            obj.x = x; obj.y = y; obj.z = z;
        end
        
        %constructs 3x1 vector from obj.x, obj.z coordinates,
        % can use R and T methods
        function out = vectorXYZ(obj)
            out = [obj.x;obj.y;obj.z;1];
        end
        
        % @param StepperMotor stepperMotor
        % @param double[] theta, array of projection angles
        % @param varargin - only length 1 for figure handle
        function [xC,yC,zC] = plotOrbit(obj,stepperMotor,theta,varargin)
            for i=1:length(theta)
                rotOb = stepperMotor.rotate(obj,i,theta);
                xC(i) = rotOb.getX;
                yC(i) = rotOb.getY;
                zC(i) = rotOb.getZ;
            end
            if nargin==4
                plot3(varargin{1},xC,yC,zC); xlabel('x'); ylabel('y'); zlabel('z'); axis square; drawnow;
            else
                figure; plot3(xC,yC,zC); xlabel('x'); ylabel('y'); zlabel('z'); axis square; drawnow;
            end
            
            xIdeal = (obj.x-stepperMotor.getX).*cos(theta)-(obj.z-stepperMotor.getZ).*sin(theta)+stepperMotor.getX;
            yIdeal = repmat(obj.y,1,length(theta));
            zIdeal = (obj.x-stepperMotor.getX).*sin(theta)+(obj.z-stepperMotor.getZ).*cos(theta)+stepperMotor.getZ;
            hold on; plot3(xIdeal,yIdeal,zIdeal,'r'); hold off; title('xyz object space orbit of point');
            
            figure; subplot(2,1,1); plot(xIdeal); title('x motion (mm)'); subplot(2,1,2); plot(zIdeal); title('z motion (mm)');
            
            if ~isempty(stepperMotor.getZMotion)
                figure; plot(1:length(theta),xC-xIdeal);
                hold on; plot(stepperMotor.getZMotion.*sin(theta)-stepperMotor.getXMotion.*cos(theta)+stepperMotor.getXMotion); hold off;
                xlabel('projection number'); ylabel('x displacement (mm)'); title('X-displacement from ideal sinusoid');
            end
            
        end
        
        function objOut = rotateXY(obj,theta)
            R = PointObject.Rxy(theta);
            R = R*obj.vectorXYZ;
            objOut = PointObject(R(1),R(2),obj.getZ);
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
    
    %% static methods
    methods (Static)
        % calculates rotation matrix about y axis, in xz plane (2D) by angle theta
        %@ param double theta, angle of rotation in radians (anticlock)
        function out = Rxz(theta)
            out = [cos(theta),0,-sin(theta),0;
                    0,1,0,0;
                    sin(theta),0,cos(theta),0;
                    0,0,0,1];
        end
        
        % calculates rotation matrix about z axis, in xy plane (2D) by angle theta
        %@ param double theta, angle of rotation in radians (anticlock)
        function out = Rxy(theta)
            out = [cos(theta),-sin(theta),0,0;
                    sin(theta),cos(theta),0,0;
                    0,0,1,0;
                    0,0,0,1];
        end
        
        % param double a,b,c translates by amount a,b,c
        % dimension
        function out = T(a,b,c)
            out = eye(4);
            out(1,4) = a;
            out(2,4) = b;
            out(3,4) = c;
        end
        
        % translates point P(a,b,c) to the origin
        function out = T_origin(a,b,c)
            out = eye(4);
            out(1,4) = -a;
            out(2,4) = -b;
            out(3,4) = -c;
        end
            
    end
end

