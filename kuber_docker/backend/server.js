const express=require("express")
const mongoose=require("mongoose")
const app=express()
const bodyParser=require("body-parser")
const port=process.env.PORT
const db_host=process.env.MONGO_DB_HOST
const db_name=process.env.DB_NAME
console.log(`${db_host}`)
console.log(`${db_name}`)
// handles the request to url/health
app.use(bodyParser.json());

const healthRouter = express.Router();
// health routes
healthRouter.get('/', (req, res) => {
  res.status(200).send('up!');
});
app.use('/health', healthRouter);

app.use("/store", require("./routes/store"));

module.exports = {
  healthRouter,
};
const username=process.env.MONGO_DB_USER
const password=process.env.MONGO_DB_PASSWORD

console.log("Connecting to DB")
console.log(username)
console.log(password)
mongoose.connect(
  `mongodb://${process.env.MONGO_DB_HOST}/${process.env.DB_NAME}`,
    {
      auth: {
        username: username,
        password: password,
      
    },
    connectTimeoutMS: 500
  }
)
.then(() => {
    app.listen(port, () => { 
      // wanna make sure the db is running first
    console.log(`Listening on port ${port}`);
    });
    console.log("Connected to DB");
}).catch(err =>{
  console.log("Something went wrong");
    console.log(err);
});

