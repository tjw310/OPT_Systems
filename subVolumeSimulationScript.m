% sub-volume system simulation script

sv = SubVolumeSystem;

c = ConeBeamSystem;

%%
sv.setWidth(2560);
sv.setHeight(2160);
sv.setBinFactor(4);

sv.setRotationDirection('clock');
sv.setAxisDirection('vert');
sv.setR(R);
sv.setNProj(100);
sv.setNAngles(360);

sv.stepperMotor = StepperMotor;
sv.stepperMotor.setAngle(.1);
sv.translationStage  =TranslationStage;

obTrack = PointObject(-0.2322,0,-0.065);
obCone = PointObject(-.05,0,.05);
ob = PointObject(obTrack.getX+obCone.getX,obTrack.getY+obCone.getY,obTrack.getZ+obCone.getZ);
ob = ob.rotateXY(sv.stepperMotor.getAngle);
sv.objectiveStage = ObjectiveStage;

sv.objective = Objective(.5,50);
sv.setApertureRadius(sv.objective.getRadiusPP/8);

sv.setRotBool(1);
%%
sv.objectiveStage.trackPoint(sv,obTrack);
sv.translationStage.trackPoint(sv,obTrack,1e-10);


%%
c.setHeight(2160);
c.setWidth(2560);
c.setBinFactor(sv.getBinFactor);
c.stepperMotor = sv.stepperMotor;
c.setRotationDirection('clock');
c.setAxisDirection('vert');
c.setR(R);
c.setNProj(100);
c.setNAngles(360);
c.setApertureRadius(sv.getApertureRadius);
c.setRotBool(1);
c.objective = sv.objective;

%%
obCone = PointObject(-.05,0.05,.05);
obCone = obCone.rotateXY(-sv.stepperMotor.getAngle);
obTrack = PointObject(-0.2322,0,-0.065);
ob = PointObject(obTrack.getX+obCone.getX,obTrack.getY+obCone.getY,obTrack.getZ+obCone.getZ);
sv.translationStage.trackPoint(sv,obTrack,1e-10);