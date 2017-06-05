classdef TranslationStage < handle
    %Class representing translation of the object and motor in the xy plane
    
    properties (Access = private)
        motion % double[] x-values for tracking continuously
        stepSize % double value for discrete step size
        fullMotion %double[] record of continuous motion if have discretised
    end
    
    methods 
        %constructor
        function obj = TranslationStage()
        end
        
        function out = getMotion(obj)
            out = obj.motion;
        end
        
        function plotMotion(obj)
            figure; plot(obj.motion); hold on; plot(obj.fullMotion); hold off;
        end
        
        function setStepSize(obj,stepSize)
            obj.stepSize = stepSize;
        end
        
        % fills motion with array of x-values for sinusoidal motion around
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
        % @param StepperMotor stepperMotortStattasdfewasefsd ds 
        % @param PointObject point
        function trackPoint(obj,optSys,stepperMotor,point)
            r = sqrt((point.getX-stepperMotor.getX)^2+(point.getZ-stepperMotor.getZ)^2);
            phi = atan2(point.getZ-stepperMotor.getZ,point.getX-stepperMotor.getX);
            obj.fullMotion = r.*cos(phi+optSys.theta);
            if ~isempty(obj.stepSize) 
                obj.discretiseMotion;
            else
                obj.motion = obj.fullMotion;
            end
            obj.plotMotion;
        end
        
        % @param OPTSystem optSys
        % @param StepperMotor stepperMotor
        % @param PointObject point
        function discretiseMotion(obj)
            if isempty(obj.stepSize) || isempty(obj.motion)
                error('Please assign a step size and ideal tracking motion');
            else
                numberSteps = round(obj.fullMotion./obj.stepSize);
                obj.motion = numberSteps.*obj.stepSize;
            end
        end
        
        % @param int projectionNumber, projection number
        % @return double out, x-position of tStage in mm
        function out = getMotionAtProj(obj,projectionNumber)
            if isempty(obj.motion)
                out = 0;
            else
                out = obj.motion(projectionNumber);
            end
        end
        
        % @return double[] out, difference between current motion profile
        % and the full continuous motion profile
        function out = discreteDifference(obj)
           out = obj.motion-obj.fullMotion;
        end
           
    end
    
end

