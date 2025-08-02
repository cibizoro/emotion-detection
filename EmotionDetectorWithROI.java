import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.core.Rect;

import org.opencv.highgui.HighGui;

import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.util.Collections;

public class EmotionDetectorWithROI {

    static {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static final String[] EMOTION_LABELS = {
        "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
    };

    public static void main(String[] args) throws Exception {

        // Update this path to your actual XML file path
        String faceCascadePath = "C:\\Users\\CIBI ARULNATH\\cibicv\\cibicv10\\cibicv10\\resources\\haarcascade_frontalface_default.xml";

        CascadeClassifier faceDetector = new CascadeClassifier(faceCascadePath);
        if (faceDetector.empty()) {
            System.err.println("Failed to load cascade classifier at " + faceCascadePath);
            return;
        }

        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("Cannot open webcam");
            return;
        }

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        // Update this path to your ONNX model path
        OrtSession session = env.createSession("C:\\Users\\CIBI ARULNATH\\cibicv\\cibicv10\\cibicv10\\models\\emotion_model.onnx", opts);

        Mat frame = new Mat();

        while (true) {
            if (!camera.read(frame) || frame.empty()) {
                System.out.println("No frame captured");
                continue;
            }

            // Flip frame for mirror effect
            Core.flip(frame, frame, 1);

            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(frame, faces);

            for (Rect face : faces.toArray()) {
                // Draw rectangle around face
                Imgproc.rectangle(frame, face, new Scalar(0, 255, 0), 2);

                Mat faceROI = new Mat(frame, face);

                // Resize face ROI to 48x48 (model input size)
                Imgproc.resize(faceROI, faceROI, new Size(48, 48));

                // Convert to float and normalize pixel values [0..1]
                faceROI.convertTo(faceROI, CvType.CV_32FC3, 1.0 / 255.0);

                // Prepare input tensor in NHWC format
                float[] inputData = new float[48 * 48 * 3];
                int idx = 0;
                for (int y = 0; y < 48; y++) {
                    for (int x = 0; x < 48; x++) {
                        double[] pixel = faceROI.get(y, x);
                        for (int c = 0; c < 3; c++) {
                            inputData[idx++] = (float) pixel[c];
                        }
                    }
                }

                try (OnnxTensor inputTensor = OnnxTensor.createTensor(env,
                        FloatBuffer.wrap(inputData), new long[]{1, 48, 48, 3})) {

                  OrtSession.Result output = session.run(Collections.singletonMap("input", inputTensor));


                    float[][] prediction = (float[][]) output.get(0).getValue();

                    int maxIdx = 0;
                    for (int i = 1; i < prediction[0].length; i++) {
                        if (prediction[0][i] > prediction[0][maxIdx]) maxIdx = i;
                    }

                    String emotion = EMOTION_LABELS[maxIdx];

                    Imgproc.putText(frame, emotion, new Point(face.x, face.y - 10),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0), 2);
                }
            }

            HighGui.imshow("Emotion Detection", frame);
            if (HighGui.waitKey(1) == 27) { // ESC key to quit
                break;
            }
        }

        camera.release();
        HighGui.destroyAllWindows();
    }
}
//comile javac -cp ".;../lib/opencv-4110.jar;../lib/onnxruntime-1.17.0.jar" EmotionDetectorWithROI.java
// run java -Djava.library.path="../lib" -cp ".;../lib/opencv-4110.jar;../lib/onnxruntime-1.17.0.jar" EmotionDetectorWithROI

