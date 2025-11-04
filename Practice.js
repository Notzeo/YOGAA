  import { poseLabels } from "./pose_labels.js";

  let video;
  let poseNet;
  let pose;
  let skeleton;
  let poseLabel;
  let brain;
  let state = 'waiting';
  let thirtysecs;
  let posesArray = ['TADASANA',  'VIRABHADRASANA I','VIRABHADRASANA II','VRIKSHASANA','TRIKONASANA','Adho Mukho Sawasana'];
  var imgArray = new Array();


  var poseImage;

  var targetLabel;
  var errorCounter;
  var iterationCounter;
  var poseCounter;
  var target;

  var timeLeft;
  var env;
  var wave;


  function setup() {
    var videoContainer = document.getElementById("videoContainer");
    var car = createCanvas(600, 470);
    car.parent(videoContainer); // this attaches the canvas inside the div from HTML

    video = createCapture(VIDEO)
    video.hide();
    poseNet = ml5.poseNet(video, modelLoaded);
    poseNet.on('pose',gotPoses);

    env =  loadSound("images/file.mp3");
    wave = loadSound("images/error.mp3");
    imgArray[0] = new Image();
    imgArray[0].src = "images/urdhava.jpg";
    imgArray[1] = new Image();
    imgArray[1].src = "images/warrior1.gif";
    imgArray[2] = new Image();
    imgArray[2].src = "images/warrior2.gif";
    imgArray[3] = new Image();
    imgArray[3].src = "images/Tree.gif";
    imgArray[4] = new Image();
    imgArray[4].src = "images/Tri.gif";
    imgArray[5] = new Image();
    imgArray[5].src = "images/adhomukh.gif";

    poseCounter = 0;
    targetLabel = 1;
    target = posesArray[poseCounter];
    document.getElementById("poseName").textContent = target;
    timeLeft = 30;
    document.getElementById("time").textContent = "00:" + timeLeft;
    errorCounter = 0;
    iterationCounter = 0;
    document.getElementById("poseImg").src = imgArray[poseCounter].src;
    
    
    let options = {
      inputs: 34,
      outputs:6,
      task:'classification',
      debug: true
    }
    
    brain = ml5.neuralNetwork(options);
    const modelInfo = {
      model: 'model/model.json',
      metadata: 'model/model_meta.json',
      weights: 'model/model.weights.bin',
    };
    brain.load(modelInfo,brainLoaded);
  }

  function brainLoaded(){
    console.log("Model ready!");
    classifyPose();
  }

  function classifyPose(){
    if (pose) {
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
      brain.classify(inputs, gotResult);
    } else {
      console.log("Pose not found");
      setTimeout(classifyPose, 500);
    }
  }

  function gotResult(error, results) {
    if (error) {
      console.error("Classification error:", error);
      setTimeout(classifyPose, 500);
      return;
    }

    if (!results || results.length === 0) {
      console.warn("No results returned");
      setTimeout(classifyPose, 500);
      return;
    }

    const { label, confidence } = results[0];
    document.getElementById("welldone").textContent = "";
    document.getElementById("sparkles").style.display = "none";

    // Only trust confident predictions
    if (confidence > 0.75) {
      console.log("Confidence", label);

      // Correct pose detected
      if (label === targetLabel.toString()) {
        iterationCounter++;

        if (iterationCounter >= 30) {
          iterationCounter = 0;
          env.play();
          nextPose();
        } else {
          timeLeft--;
          document.getElementById("time").textContent = `00:${String(timeLeft).padStart(2, "0")}`;
          setTimeout(classifyPose, 700);
        }

      // Wrong pose detected
      } else {
        errorCounter++;
        console.log("error");

        // Only trigger after 4 continuous wrong frames
        if (errorCounter >= 4) {
          console.warn("four errors - resetting timer");
          errorCounter = 0;
          iterationCounter = 0;
          timeLeft = 30;
          wave.play();
        }
        setTimeout(classifyPose, 700);
      }

    }
    const videoContainer = document.getElementById("videoContainer");

  if (confidence > 0.75) {
    console.log("Confidence", label);

    if (label === targetLabel.toString()) {
      videoContainer.classList.add("correct-glow");
      videoContainer.classList.remove("incorrect-glow");
      iterationCounter++;

      if (iterationCounter >= 30) {
        iterationCounter = 0;
        env.play();
        nextPose();
      } else {
        timeLeft--;
        document.getElementById("time").textContent = `00:${String(timeLeft).padStart(2, "0")}`;
        setTimeout(classifyPose, 700);
      }

    } else {
      videoContainer.classList.add("incorrect-glow");
      videoContainer.classList.remove("correct-glow");
      errorCounter++;
      console.log("error");

      if (errorCounter >= 4) {
        console.warn("four errors - resetting timer");
        errorCounter = 0;
        iterationCounter = 0;
        timeLeft = 30;
        wave.play();
      }
      setTimeout(classifyPose, 700);
    }
  } 
    else {
      console.log("Low confidence - skipping");
      setTimeout(classifyPose, 700);
    }
  }


  function dataReady(){
    brain.normalizeData();
    brain.train({epochs:100},finished );
  }

  function finished(){
      console.log('model trained');
    brain.save();
  }

  function gotPoses(poses){
    
    if(poses.length > 0){
      pose = poses[0].pose;
      skeleton = poses[0].skeleton;
      
      if(state=='collecting'){
      
      let inputs = [];
      for(let i =0 ;i < pose.keypoints.length; i++){
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
          inputs.push(y); 
      }
      let target = [targetLabel];
      brain.addData(inputs,target );
    }
      
  }  
  }

  function modelLoaded(){
    console.log('poseNet Ready');
  }
  function draw() {
    push();
    translate(video.width,0)
    scale(-1,1)
    image(video ,0,0,video.width,video.height);
    
    if(pose){
      for(let i =0 ;i < pose.keypoints.length; i++){
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        fill(0,255,0)
        ellipse(x,y,16,16);
        
      }
      for(let i =0 ;i < skeleton.length; i++){
        let a = skeleton[i][0];
        let b = skeleton[i][1];
        strokeWeight(6);
        stroke(257)
        line(a.position.x,a.position.y,b.position.x,b.position.y);
      }
    
      fill(255,0,255);
      noStroke();
      textSize(256);
      textAlign(CENTER,CENTER);
      text(poseLabel,width/2,height/2);
    }
    pop();
  }

  function nextPose(){

    if (poseCounter >= 5) {
      console.log("Well done, you have learnt all poses!");
      document.getElementById("welldone").textContent = "All poses done.";
      document.getElementById("sparkles").style.display = 'block';
      document.getElementById("poseName").style.display = 'none';
      document.getElementById("poseImg").style.display = 'none';
      document.getElementById("time").style.display = 'none';
      document.getElementById("sec").style.display = 'none';

    }else{
      //var stars = document.getElementById("starsid");
      //stars.classList.add("stars.animated");
      errorCounter = 0;
      iterationCounter = 0;
      poseCounter = poseCounter + 1;
      targetLabel = poseCounter + 1;
      console.log("next pose target label" + targetLabel)
      target = posesArray[poseCounter];
      document.getElementById("poseName").textContent = target;
      document.getElementById("welldone").textContent = "Well done, next pose!";
      document.getElementById("sparkles").style.display = 'block';
      document.getElementById("poseImg").src = imgArray[poseCounter].src;
      console.log("classifying again");
      timeLeft = 30;
      document.getElementById("time").textContent = "00:" + timeLeft;
      setTimeout(classifyPose, 4000)}
  } 