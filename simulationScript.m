%script to set up cone beam system with simulation 19th May 2017. T.Watson

%system = SimpleConeRecon();
% NA = 0.4; mag = 20; axialDisplacement = -150; apertureReductionFactor = 10; lambda = 500e-6;
% objective = Objective(0.4,-20);
% object = PointObject(0.1,-0.25,0);
% coneSystem = ConeBeamSystem(objective.getRadiusPP/apertureReductionFactor,lambda);
% 
% system.setWidth(1048);
% system.setHeight(2110);
% system.setAxisDirection('vert');
% system.setRotationDirection('anti');
% system.setNAngles(360);
% system.setNProj(128);
% system.setR(10000);
% system.setBinFactor(2);
% 
% 
% system.simulateProjections(objective,object,coneSystem);

%system.reconPS(object.getY,object.getY)

%system = ConeBeamSystem();
NA = 0.4; mag = 20; apertureReductionFactor = 3; lambda = 500e-6;
objective = Objective(NA,mag);
%object = [PointObject(0.1,0.1,0.1),PointObject(0,0.1,0),PointObject(0,0.1,-0.1),PointObject(0.1,0.1,0),PointObject(-0.1,0.1,-0.1)];

object = [PointObject(.05,0.05,0),PointObject(.05,0,0),PointObject(0,0,0)];

system.setLambda(500e-6);
system.setWidth(512);
system.setHeight(512);
system.setAxisDirection('vert');
system.setRotationDirection('anti');
system.setNAngles(360);
system.setNProj(400);
system.setBinFactor(1);
system.setApertureRadius(objective.getRadiusPP/apertureReductionFactor);
system.setR(6000);

%system2 = Standard4fSystem();
system2.setLambda(system.getLambda);
system2.setWidth(system.getWidth);
system2.setHeight(system.getHeight);
system2.setAxisDirection(system.getAxisDirection);
system2.setRotationDirection(system.getRotationDirection);
system2.setNAngles(system.getNAngles);
system2.setNProj(system.getNProj);
system2.setBinFactor(system.getBinFactor);
system2.setApertureRadius(system.getApertureRadius);
%system.simulateProjections(objective,object,1);
%%
%round(object(1).getYPixelPosition(system,objective))
%system.reconstructProjections(round(object(1).getYPixelPosition(system,objective)),round(object(1).getYPixelPosition(system,objective)),1);



