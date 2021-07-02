import cv2
import numpy as np

def neural_network_train(self,input_cols,output_cols,filename,model_name,epochs,batch_size,delimiter,condition_name, target_accuracy = 0.95):
        try:
                
            self.input_cols = input_cols
            self.output_cols = output_cols
            self.filename = filename
            self.model_name = model_name
            self.epochs = epochs
            self.batch_size = batch_size
            self.delimiter =delimiter
            self.condition_name=condition_name
            self.model_accuracy = 0.0
            
        

            # if not os.path.exists(f"./static/shap_plots/{self.condition_name}"):
                # os.mkdir(f"./static/shap_plots/{self.condition_name}")

            inputs = list(map(str, input_cols.split(',')))
            # if comma in output_cols:

            outputs = list(map(str,output_cols.split(",")))
            if len(inputs)%2==0 :
                self.max_tries =len(inputs)//2
            else: 
                self.max_tries =len(inputs)//2 +1
            # else:
            # outputs = output_cols
            iteration = 0
            flag=0
            drop_col = ""
            while (self.model_accuracy < target_accuracy) and len(inputs) >= 2 and self.max_tries!=0 :
                print("################################## MAX_TRIES ######################################## " ,self.max_tries)
                self.max_tries-=1
                
                df=pd.read_csv(f"./static/nn_csv/{filename}", delimiter = delimiter)
                print(df.shape)
                
                if flag==0:
                    X = df[inputs] 
                    X['Extra'] = 123 # We have added an extra column with constant values that doesn't support the target variable...
                    y = df[outputs]
                column_to_drop = drop_col
                if drop_col is not "":
                    X.drop(column_to_drop,inplace=True, axis=1)
                X_df=X
                X_train_cols=X.columns
                print(X.dtypes,y.dtypes)
                print("These are inputs : ",X.columns)
                print("These are outputs : ",y.columns)
                print(X.shape)
                print(X.head())

                # X=X.values
                # y=y.values
                n_cols=X.shape[1]
                

                # st_scaler = preprocessing.StandardScaler()
                # X_scale = st_scaler.fit_transform(X)
                

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                print("SHAPES : ",X_train.shape, X_test.shape, y_train.shape, y_test.shape)
                # mean = X_train.mean(axis=0)
                # std = X_train.std(axis=0)

                # X_train = (X_train - mean) / std
                # X_test = (X_test - mean) / std

                # print(X_train[0].shape())
                model = Sequential()
                model.add(Dense(128, input_shape=(n_cols,), activation= "relu"))
                model.add(Dense(64, activation= "relu"))
                model.add(Dense(32, activation= "relu"))
                # model.add(Dense(16, activation= "relu",kernel_initializer="he_normal",bias_initializer='zeros'))
                model.add(Dense(8, activation= "relu"))
                model.add(Dense(1,activation= "linear"))
                optimizer = keras.optimizers.Adam(lr=0.0001)
                model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae','accuracy'])

                history = model.fit(X_train, y_train, epochs=epochs ,batch_size=batch_size,shuffle=True, verbose=1)

                model_path = f"./static/models/{model_name}" 
                model.save(model_path)
            
                print(f"Training Iteration {iteration} Completed")
                
                score = model.evaluate(X_test, y_test, verbose = 0) 

                print('Test loss:', score[0]) 
                print('Test accuracy:', score[1])
                self.model_accuracy = score[1]
                
                

                explainer = shap.KernelExplainer(model.predict, X_train)
                shap_values = explainer.shap_values(X_test,nsamples=100)
                # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
                shap.summary_plot(shap_values = shap_values, features = X_test,feature_names = X_train_cols, show=False, plot_type='bar')
                plt.savefig(f'./static/shap_plots/{condition_name}/Iteration_{iteration}_{self.model_accuracy}.png',dpi=1200, bbox_inches='tight')
                
                vals= np.abs(shap_values[0]).mean(0)
                print("VALS.....",vals)

                out_features = X_df.columns[np.argsort(np.abs(shap_values[0]).mean(0))]
                print(out_features)
                print(f"Column to be dropped in Iteration ---{iteration}",out_features[0])
                drop_col = out_features[0]

                ######  CORRELATION..
                # corr_matrix = X_df.corr().abs()
                # print(corr_matrix)

                # # Select upper triangle of correlation matrix
                # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

                
                # # Find features with correlation greater than 0.95
                # to_drop = [column for column in upper.columns if any(upper[column] > 0.50)]
                # print("DROP COLS : ",to_drop)
                # print(f"Column to be dropped in Iteration ---{iteration}",to_drop[0])
                # drop_col =to_drop[0]
                # # Drop features 
                # # df.drop(to_drop, axis=1, inplace=True)

                iteration += 1
                flag=1
                

            print("All Training Iterations Completed. Optimised model saved.")
        

            return model_path
        except Exception as e:
            print(e)