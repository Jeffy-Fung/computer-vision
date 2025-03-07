from ultralytics import YOLO, checks, hub

def main():
  checks()

  hub.login('090a5d8d42823c97e81ed7b14d1e46615a78eb7307')

  model = YOLO('https://hub.ultralytics.com/models/kC4aihLdb0XJou7YxvoK')
  results = model.train()
  results = model.val()
  results = model.export(format='onnx')

if __name__ == '__main__':
  main()