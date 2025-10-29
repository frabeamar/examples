const keyValueDb = process.env.DB_NAME;
const keyValueUser = process.env.MONGO_DB_USER;
const keyValuePassword = process.env.MONGO_DB_PASSWORD;

console.log("parameters");
console.log(keyValueDb);
console.log(keyValueUser);
console.log(keyValuePassword);
db = db.getSiblingDB(keyValueDb);

db.createUser({
  user: keyValueUser,
  pwd: keyValuePassword,
  roles: [
    {
      role: 'readWrite',
      db: keyValueDb,
    },
  ],
});
