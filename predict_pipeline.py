import sys
import pandas as pd
import pickle
import dill
      
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            #logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            print('Exception Occured in prediction pipeline',e)
            #raise CustomException(e,sys)
            
            
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            filename = open('./Gemstone/data/gemsmodel.pkl', 'rb')
            model = pickle.load(filename)
            filename.close()

            filename1 = open('./Gemstone/data/prerocessor.pkl', 'rb')
            preprocessor = pickle.load(filename1)
            filename1.close()
            
            data_scaled = preprocessor.fit_transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            print('Exception occured in prediction pipeline',e)
            #raise CustomException(e,sys)






'''    
    def load_object(file_path):
        try:
            with open(file_path,'rb') as file_obj:
                return dill.load(file_obj)
        except Exception as e:
            print('Exception occured in prediction pipeline',e)
    '''
'''
preprocessor_path = './Gemstone/data/prerocessor.pkl'
model_path = './Gemstone/data/gemsmodel.pkl'
preprocessor = load_object(file_path=preprocessor_path)
model = load_object(file_path=model_path)
'''