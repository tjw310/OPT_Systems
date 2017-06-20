classdef TranslationStage < handle
    %Class representing translation of the object and motor in the xy plane
    
    properties (Access = private)
        motion % double[][] x/y-values for tracking continuously
        stepSize =[0;0] % double[] value for discrete step size in x, or x&y if 2D
        fullMotion %double[][] x/y record of continuous motion if have discretised
        angle %angle of translation stage axis with respect to detector pixels
        magAoR %magnification of movements at the axis of rotation. i.e dx(pixels)=dx(mm)*optSystem.PixelSize/TranslationStage.magAoR;
        motionAngle %angle of motion in radians, motion should be linear at an angle
        type % string, either discrete or continuous
        
        motionPixels %double [1x2][] discrete stage motion in image-space binned pixels
        fullMotionPixels %double [1x2][] continuous stage motion in image-space binned pixels
    end
    
    methods 
        %% CONSTRUCTOR
        function obj = TranslationStage()
        end
        
        %% GET /SET
        % @param string type, type of motion either 'discrete' or
        % 'continuous'
        function setType(obj,type)
            switch type
                case 'discrete'
                    obj.type = 'discrete';
                case 'continuous'
                    obj.type = 'continuous';
                otherwise
                    error('Incorrect stage motion type');
            end
        end  
        % @param double magAoR, magnifcation at the axis of rotation
        function setMagAoR(obj,magAoR)
            obj.magAoR = magAoR;
        end
        % @param double stepSize, size of discrete steps in mm (can be 1D
        % or 2D depending on if x or x&y motion)
        function setStepSize(obj,stepSize)
            obj.stepSize = stepSize;
        end
        % @param double[] motion, 1D array of motion
        function setMotion(obj,motion)
            obj.motion = motion;
        end
         % @param double[] motion, 1D array of continuous motion
        function setFullMotion(obj,fullMotion)
            obj.fullMotion = fullMotion;
        end
        % @param double motionAngle, angle of linear motion in xy plane
        function setMotionAngle(obj,motionAngle)
            obj.motionAngle = motionAngle;
        end
        % @param double angle, angle of translation stage axis in relation
        % to image pixels
        function setAngle(obj,angle)
            obj.angle = angle;
        end
        function out = getType(obj)
            out = obj.type;
        end
        function out = getStepSize(obj)
            out = obj.stepSize;
        end
        function out = getAngle(obj)
            out = obj.angle;
        end
        function out = getMagAoR(obj)
            out = obj.magAoR;
        end        
        function out = getMotion(obj)
            out = obj.motion;
        end
        function out = getFullMotion(obj)
            out = obj.fullMotion;
        end
        function out = getMotionAngle(obj)
            out = obj.motionAngle;
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
        
        % @param int projectionNumber, projection number
        % @param OPTSystem optSys
        % @return double out, difference betweeen discrete and continuous
        % motion position of tStage in image-space pixels for X. outputs
        % full discrete y-motion
        function out = getDiscreteDifferencePixels(obj,optSys)
             [discrete,full] = getAllMotionPixels(obj,optSys);
             out(1,:) = discrete(1,:)-full(1,:);
             out(2,:) = discrete(2,:)-mean(full(2,:));
        end
        
        % @param int projectionNumber, projection number
        % @param OPTSystem optSys
        % @return double out, x-position of tStage in image-space pixels
        function out = getMotionPixels(obj,projectionNumber,optSys)
            if isempty(obj.motionPixels)
                allDiscreteMotion = obj.getAllMotionPixels(optSys);
            else
                allDiscreteMotion = obj.motionPixels;
            end
            out = allDiscreteMotion(:,projectionNumber);      
        end
        
        % @param int projectionNumber, projection number
        % @param OPTSystem optSys
        % @return double out, x-position of tStage in image-space pixels
        function [discrete,full] = getAllMotionPixels(obj,optSys)
            if isempty(obj.fullMotion)
                error('please set motion of translation stage');
            else
                if isempty(obj.magAoR)
                    mag = optSys.objective.getMagnification;
                else
                    mag = obj.magAoR;
                end
                fullMotion = obj.fullMotion;
                av = repmat(mean(fullMotion,2),1,size(fullMotion,2));
                fullMotion = fullMotion-av;
                step = obj.stepSize;
                
                if ~isempty(obj.angle)
                    if optSys.getRotBool==1
                        theta = obj.angle+optSys.stepperMotor.getAngle;
                    else
                        theta = obj.angle;
                    end
                else
                    if optSys.getRotBool==1
                        theta = optSys.stepperMotor.getAngle;
                    else
                        theta = 0;
                    end
                end
                
                if theta~=0
                    fullMotion = TranslationStage.rotateXYvector(fullMotion,theta);
                    av = TranslationStage.rotateXYvector(av,theta);
                    step = TranslationStage.rotateXYvector(step,theta);
                end
                
                full = (fullMotion+av)/optSys.getPixelSize*mag;
                if strcmp(obj.type,'discrete')
                    discrete = TranslationStage.discretise(fullMotion+av,step)/optSys.getPixelSize*mag;
                else
                    discrete = full;
                end
                full(isnan(full)) = 0; discrete(isnan(discrete)) = 0;
                obj.motionPixels = discrete;
                obj.fullMotionPixels = full;
            end
        end
        
        function clearMotionPixels(obj)
            obj.motionPixels = [];
        end
        
         
        %% PLOT
        function plotMotion(obj)
             figure; 
            for k=1:size(obj.motion,1)
                subplot(1,size(obj.motion,1),k)
                plot(obj.motion(k,:)); hold on; plot(obj.fullMotion(k,:)); hold off; 
                xlabel('Projection Number');
                ylabel('Distance (mm)'); axis square;
                switch k
                    case 1
                        title('x Motion');
                    case 2
                        title('y Motion');
                end
            end
        end
        
        function plotMotionPixels(obj)
            if ~isempty(obj.motionPixels)
                 figure; 
            for k=1:size(obj.motionPixels,1)
                subplot(1,size(obj.motionPixels,1),k)
                plot(obj.motionPixels(k,:)); hold on; plot(obj.fullMotionPixels(k,:)); hold off; 
                xlabel('Projection Number');
                ylabel('Distance (pixels)'); axis square;
                switch k
                    case 1
                        title('x Motion');
                    case 2
                        title('y Motion');
                end
            end
            end
        end
        
        %% SIMULATION
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
        function trackPoint(obj,optSys,point,varargin)
            stepperMotor = optSys.stepperMotor;
            obj.motionPixels = [];
            obj.fullMotion = zeros(2,optSys.getNProj);
            obj.motion = zeros(2,optSys.getNProj);
            for idx = 1:optSys.getNProj
                [~,x(idx),y(idx)] = stepperMotor.rotate(point,idx,optSys.theta);
            end
            if ~isempty(obj.angle) && obj.angle~=0
                theta = -obj.angle;
                obj.fullMotion = TranslationStage.rotateXYvector([x;y],theta);
            else
                obj.fullMotion(1,:) = x; obj.fullMotion(2,:) = y;
            end
            
            if ~isempty(obj.stepSize) || nargin==4
                if nargin==4
                    obj.stepSize(1)=varargin{1}*cos(stepperMotor.getAngle);
                    obj.stepSize(2)=varargin{1}*sin(-stepperMotor.getAngle);
                    if ~isempty(obj.angle) && obj.angle~=0;
                        disp('test');
                        stepSz = TranslationStage.rotateXYvector(obj.stepSize,-obj.angle);
                    else
                        stepSz = obj.stepSize;
                    end
                else
                    stepSz = obj.stepSize;
                end
                out = TranslationStage.discretise(obj.fullMotion,stepSz);
                obj.stepSize = stepSz;
                obj.motion = out;
            else
                obj.motion = obj.fullMotion;
            end
            obj.plotMotion;
        end
        
        % @param double factor, factor that will multiply the
        % y-motion (both full and discrete) of the stage by factor
        function offsetYMotion(obj,factor)
            yMotion = obj.fullMotion(2,:); stepY = obj.stepSize(2);
            yM2 = yMotion.*factor; stepY2 = stepY*factor;
            obj.fullMotion(2,:) = yM2;
            obj.stepSize(2) = stepY2;
            obj.motion = TranslationStage.discretise(obj.fullMotion,obj.stepSize);
            
            obj.plotMotion;
        end
            
        
    end
    
    %% Static Methods
    methods (Static)
        % @param double[1x2][] motion
        % @param double[1x2] stepSize
        function out = discretise(motion,stepSize)
            av = nanmean(motion,2);
            numberSteps = round((motion(1,:)-av(1))./stepSize(1));
            out(1,:) = numberSteps.*stepSize(1)+av(1);
            out(2,:) = numberSteps.*stepSize(2)+av(2);
            %subplot(1,2,1); plot(motion(1,:)); hold on; plot(out(1,:)); hold off;
            %subplot(1,2,2); plot(motion(2,:)); hold on; plot(out(2,:)); hold off; drawnow;
        end
        
        function [ estimates,continuousMotion,flag ] = fitSinusoidToDiscreteData(discreteMotion)
            if size(discreteMotion,1)~=1
                discreteMotion = discreteMotion.';
            end
            
           % discreteMotion = discreteMotion*10^3;
            
            step = discreteMotion(2:end)-discreteMotion(1:end-1);
            step(step==0)=NaN;
            step = nanmean(abs(step));

            start_guesses = [(max(discreteMotion)-min(discreteMotion))/2,-pi/2,min(discreteMotion)+(max(discreteMotion)-min(discreteMotion))/2];
            model = @fun;
            options = optimset('MaxFunEvals',5000*length(start_guesses),'MaxIter',5000*length(start_guesses));
            [estimates,~,flag] = fminsearch(model,start_guesses,options);

            % figure;
            %drawnow;

            [~,continuousMotion] = model(estimates);
              %figure;
              %plot(ydata); hold on; plot(fit,'r'); hold off; drawnow;

            function [sse,continuousMotion] = fun(params)
                A = params(1);
                B = params(2);
                C = params(3);

                theta = linspace(0,2*pi-2*pi/length(discreteMotion),length(discreteMotion));

                fit = A.*sin(theta+B);

                numberSteps = round(fit./step);

                fitDiscrete = numberSteps.*step+C;
                
                continuousMotion = fit+C;

               % plot(continuousMotion); hold on; plot(fitDiscrete); hold on; plot(discreteMotion); hold off; drawnow;
               
                sse = nansum((fitDiscrete-discreteMotion).^2);
            end   
        end
        
        %function that rotates a data array that is horizontal
        %concatanation of [x;y] vectors
        % @param double [1x2][] xyVector
        % @param double theta, angle in radians
        function xyVectorOut= rotateXYvector(xyVector,theta)
            R = [cos(theta),-sin(theta);sin(theta),cos(theta)];   
            for k=1:size(xyVector,2)
                xyPoint = xyVector(:,k);
                xyVectorOut(:,k) = R*xyPoint;
            end
        end
    end
    
end

