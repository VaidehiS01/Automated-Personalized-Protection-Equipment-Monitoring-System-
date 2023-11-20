import plyer
from ultralytics import YOLO


class_names = ('helmet', 'no-helmet', 'no-vest', 'person', 'vest')
  
    
def notification():
    plyer.notification.notify(
        title='SAFETY WARNING',
        message='Personal Protection gear not detected!',
    )

if __name__ == "__main__":
 
  model = YOLO("worker.pt") 
  results = model(source='image.jpeg',show=True,conf=0.3,save=True)
  class_id = results[0].boxes.cls.cpu().numpy().astype(int)

  # If the classes are present, send a notification.
  if 1 in class_id or 2 in class_id:
    notification()
