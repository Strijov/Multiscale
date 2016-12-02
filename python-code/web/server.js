var express = require("express");
var multer = require('multer');
var app = express();
var done = false; // !!!! Change to FALSE !!!
var PythonShell = require('python-shell');
var bodyParser = require('body-parser');

//var newFilename;

var path = require('path');

console.log("Dirname: " + __dirname)

var config;
try {
  config = require(path.resolve(__dirname+'/config.json'));
} catch (ex) {
  console.log('Warning: config.json file is not found. Using default settings...');
  config = {};
}

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
  extended: true
}));
/*Configure the multer.*/

app.use(multer({ dest: './uploads/', 
 
 rename: function (fieldname, filename) {
    vnewFilename = filename+Date.now();
    return 'file'; //filename+Date.now();
  },
onFileUploadStart: function (file) {
  console.log(file.originalname + ' is starting ...')
},
onFileUploadComplete: function (file) {
  console.log('User data uploaded to  ' + file.path)
  done=true;
}
}));

/*Handling routes.*/
app.get('/',function(req,res){
  res.sendFile(__dirname + '/index.html')
});

// app.post('/api/photo',function(req,res){
//     res.sendfile(__dirname + "/views/index.html");
// });

app.post('/api/par', function(req,res){ 
  var msg = '';
  if(done==true){
    var filename = 'uploads/file.csv'
  }else{
    var filename = 'uploads/test.csv'
  }
  console.log(req.body)
  var n_requested = req.body.n_req;
  var n_historical = req.body.n_hist;
  var train_test = req.body.train_test;
  var method = req.body.method;
  var options = {
      mode: 'text',
      // pythonPath: 'C:/Users/Anastasia/Anaconda/python',
      //pythonOptions: ['-u'],
      scriptPath: path.resolve(__dirname),
    };
  console.log(method)
  if(method=='lstm'){      
      learn_rate = req.body.learn_rate;
      n_units = req.body.lstm_units;
      n_epochs = req.body.n_epochs;
      options.args = [filename, method, n_historical, n_requested, train_test, 'lr', learn_rate, 'n_epochs', n_epochs, 'n_units', n_units];
      }
  if(method=='lasso'){    
      alpha = req.body.alpha;
      options.args = [filename, method, n_historical, n_requested, train_test, 'alpha', alpha];      
  }  
  if(method=='rf'){    
      n_trees = req.body.n_trees;
      options.args = [filename, method, n_historical, n_requested, train_test, 'n_estimators', n_trees];      
  }
  if(method=='gbr'){    
      n_trees = req.body.n_trees;
      options.args = [filename, method, n_historical, n_requested, train_test];      
  }
 
    console.log(filename)
    if (config.pythonPath)
      options.pythonPath = config.pythonPath;

    //var py_func = new PythonShell('../test.py', options);

    PythonShell.run('../demo_for_web.py', options, function (err, results) {
      if (err) throw err;
      console.log('results: %j', results);  
      res.set('Content-Type', 'html');
      res.sendFile(__dirname + "/res.html")
    });
    
  
});

  
/*Run the server.*/
app.listen(process.env.EXPRESS_PORT || 3000, function(){
    console.log("Working on port " + process.env.EXPRESS_PORT || 3000);
});








