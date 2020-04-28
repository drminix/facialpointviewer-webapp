const video = document.getElementById('video');
const actionBtn = document.getElementById('button_capture');
const width = 400;
const height = 400;
const FPS = 60;
let stream;
let streaming = false;
let faceCascadeFile = 'haarcascade_frontalface_default.xml';
const LEFT_EYEBROW_OUTER_END = [14,15];
const RIGHT_EYEBROW_OUTER_END = [18,19];
const MOUSE_CENTER_BOTTOM_LIP = [28,29];
const MOUSE_CENTER_TOP_LIP = [26,27];
const MOUSE_LEFT_CORNER = [22,23];
const MOUSE_RIGHT_CORNER = [24,25];
const NOSE_TIP = [20,21];
const TARGET_IMAGE_WIDTH = 96;
const TARGET_IMAGE_HEIGHT = 96;

//for debugging with nginx
//const model_url = `http://localhost:8080/model.json`
const model_url=`https://raw.githubusercontent.com/drminix/models/master/facialpointdetection/tensorflowjs_model/model.json`

let model;
let model_loaded = false;

document.body.classList.add("loading");

function onOpenCvReady() {
  cv.onRuntimeInitialized = onReady;
  document.body.classList.remove("loading");
  console.log("OpenCV is ready: ",cv);
}

async function loadModel() {
    model = await tf.loadLayersModel(model_url);
    if(model) {
        document.getElementById("model-status").innerHTML = "Model loaded successfully";
    }
    model_loaded = true;
    
}
function meme() {
    console.log("files are loaded");
}
function onReady() {
    let src;
    let dst;
    let gray;
    const cap = new cv.VideoCapture(video);
    let classifier;
    let faces 
  
    
    actionBtn.addEventListener('click', () => {
        if (streaming) {
            stop();
            actionBtn.textContent = 'Start';
        } else {
            start();
            actionBtn.textContent = 'Stop';
        }
    });
    
    function start () {
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(_stream => {
            stream = _stream;
            console.log('stream', stream);
            video.srcObject = stream;
            video.play();
            streaming = true;
            src = new cv.Mat(height, width, cv.CV_8UC4);
            dst = new cv.Mat(height, width, cv.CV_8UC4);
            faces = new cv.RectVector();
            gray = new cv.Mat();
            
            // load pre-trained classifiers
            classifier = new cv.CascadeClassifier();
            if(classifier.load(faceCascadeFile) != true) {
                alert("error while loading face detection file");
            }
            
            setTimeout(processVideo, 0)
        })
        .catch(err => console.log(`An error occurred: ${err}`));
    }

    function stop () {
        if (video) {
            video.pause();
            video.srcObject = null;
        }
        if (stream) {
            stream.getVideoTracks()[0].stop();
        }
        streaming = false;
    }

    function processVideo () {
        if (!streaming) {
            src.delete();
            dst.delete();
            gray.delete();
            faces.delete();
            classifier.delete();
            return;
        }
        const begin = Date.now();
        cap.read(src);
        src.copyTo(dst);
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY,0);
        
        // detect faces.
        classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
        // draw faces.
        for (let i = 0; i < faces.size(); ++i) {
            let face = faces.get(i);
           
            
            let point1 = new cv.Point(face.x, face.y);
            let point2 = new cv.Point(face.x + face.width, face.y + face.height);
            cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
            
            //rect = new cv.Rect(100, 100, 200, 200);
            //dst = src.roi(rect);
            //let rect = cv.Rect(point1.x,point1.y, point2.x,point2.y);
            //if(face.width < TARGET_IMAGE_WIDTH || face.height < //TARGET_IMAGE_HEIGHT) {
            //    continue;
            //}
            if(model_loaded) {
                //convert data to array(TODO: must be a better way to copy out the data)
                const target_array = [];
                for(let row=face.y;row<face.y+TARGET_IMAGE_HEIGHT;row++) {
                    for(let col=face.x; col<face.x+TARGET_IMAGE_WIDTH;col++) {
                        let data = gray.data[row*gray.cols+col];
                        target_array.push(data);
                    }
                }

                //create tensor
                const inputTensor = tf.tensor4d(target_array, [1,96,96,1]);

                //make prediction
                const outputTensor = model.predict(inputTensor);
                const outputArray = outputTensor.dataSync();
                
                tf.dispose(inputTensor);
                tf.dispose(outputTensor);
                //console.log(outputArray);
                
                scale_x = face.width / TARGET_IMAGE_WIDTH;
                scale_y = face.height / TARGET_IMAGE_HEIGHT;
                for(let i=0;i<22;i+=2) {
                    center_x = Math.floor(face.x + outputArray[i]*scale_x);
                    center_y = Math.floor(face.y + outputArray[i+1]*scale_y);
                    cv.circle(dst, new cv.Point(center_x,center_y), 2,  [255, 255, 0,255] , 1,8,0);
                }
                
                let glasses_enabled = $('#glasses_checkbox').prop('checked');
                
                
                
            }
            
        }
        
        cv.imshow('canvas_output', dst);
        
        //schedule next call
        const delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
    }
}

loadModel();