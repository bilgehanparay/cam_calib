import numpy as np
import cv2 as cv

def imgLog(img, text, x, y):
    font                   = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x, y)
    fontScale              = 0.4
    fontColor              = (255,0,0)
    thickness              = 1
    lineType               = 2
    return cv.putText(img,text, 
                      bottomLeftCornerOfText, 
                      font, 
                      fontScale,
                      fontColor,
                      thickness,
                      lineType)

if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    columns = 6
    rows = 9
    gridSize = 33 # mm
    objp = np.zeros((columns*rows, 3), np.float32)
    objp[:,:2] = gridSize*np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    while True:
        # Capture frame by frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found_board, found_corners = cv.findChessboardCorners(gray, (columns, rows), flags=cv.CALIB_CB_ADAPTIVE_THRESH )
        if found_board:
            objpoints.append(objp)
            flog = "Found ChessBoard"
            winSize = (11,11)
            zeroZone = (-1, -1)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
            found_corners = cv.cornerSubPix(gray, found_corners, winSize, zeroZone, criteria)
            cv.drawChessboardCorners(gray, (columns, rows), found_corners, found_board)
            imgpoints.append(found_corners)
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
                mean_error += error
            print( "total error: {}".format(mean_error/len(objpoints)) )
        else:
            flog = "Not Found ChessBoard"

        gray = imgLog(gray, flog, 480, 470)
        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()