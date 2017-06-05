classdef ObjectiveStage < handle
    %Class representing translation of the optical system along the z-axis
    
    properties (Access = private)
        motion % double[] z-values for tracking, length = OPTSystem.getNProj
    end
    
    methods 
        %constructor
        function obj = ObjectiveStage()
        end
        
        function out = getMotion(obj)
            out = obj.motion;
        end
        
        function plotMotion(obj)
            figure; plot(obj.motion);
        end
        
        % fills motion with array of z-values for sinusoidal motion around
        % centre of OPTSystem volume
        % @param OPTSystem optSys, theta values
        % @param double amplitude, amplitude of motion in mm (object space)
        % @param double phase, phase of motion in radians
        % @param double varargin, optional offset of motion in mm
        function sinusoidalMotion(obj,optSys,amplitude,phase,varargin)
            if nargin==5
                offset = varargin{1};
            else
                offset = 0;
            end
            obj.motion = amplitude.*sin(optSys.theta+phase)+offset;
        end
        
        % @param OPTSystem optSys
        % @param StepperMotor stepperMotor
        % @param PointObject point
        function trackPoint(obj,optSys,stepperMotor,point)
            r = sqrt((point.getX-stepperMotor.getX)^2+(point.getZ-stepperMotor.getZ)^2);
            phi = atan2(point.getZ-stepperMotor.getZ,point.getX-stepperMotor.getX);
            obj.motion = r.*sin(phi+optSys.theta);
            obj.plotMotion;
        end
        
        % @param int projectionNumber, projection number
        % @return double out, z-position of objective stage in mm
        function out = getMotionAtProj(obj,projectionNumber)
            if isempty(obj.motion)
                out = 0;
            else
                out = obj.motion(projectionNumber);
            end
        end
           
    end
    
end

