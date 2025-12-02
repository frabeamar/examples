const express=require("express");
const router=express.Router();

router.post("/", (req, res) =>{
    return res.send("Creating key value pari")
});
router.get("/", (req, res) =>{
    return res.send("Getting key value pari")
})
router.put("/", (req, res) =>{
    return res.send("Updating key value pari")
});
router.delete("/", (req, res) =>{
    return res.send("Deleting key value pari")
});
module.exports=router;
