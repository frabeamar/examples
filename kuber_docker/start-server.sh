export MONGODB_IMAGE="mongo"
export MONGODB_TAG="8"
export BACKEND_PORT=27017
export DB_PORT=3000 
export CONTAINER_NAME="mongodb"
export VOLUME_NAME="volume_db"
export NETWORK_NAME="network_db"


source .env
if [ "$(docker ps -q -f name=$CONTAINER_NAME )" ]; then
    echo A container already exists. Killing and restarting 
    docker kill $CONTAINER_NAME
fi
echo start
# Create volume and networks
# volumes are persistant data on the host handled by docker
if [ "$(docker volume ls -q -f name=$VOLUME_NAME)" ]; then 
    echo "A  volume already exists"
    docker volume rm $VOLUME_NAME
fi

docker volume create $VOLUME_NAME

if [ "$(docker network ls -q -f name=$NETWORK_NAME)" ]; then 
    echo "Network $NETWORK_NAME already exists"
else
    docker network create $NETWORK_NAME
fi


# MONGO_INITDB_ROOT_USERNAME -> env variable from docker image
# MONGO_DB_USER is something I set
# define a volume or the database fails to start
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo killing container $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

    # -e MONGO_INITDB_ROOT_USERNAME=$ROOT_USERNAME \
    # -e MONGO_INITDB_ROOT_PASSWORD=$ROOT_PASSWORD \

if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo killing container $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

docker run  -d --name $CONTAINER_NAME \
    --network $NETWORK_NAME \
    -e DB_NAME=$DB_NAME \
    -e MONGO_DB_USER=$MONGO_DB_USER \
    -e MONGO_DB_PASSWORD=$MONGO_DB_PASSWORD \
    -v $VOLUME_NAME:/data/db \
    -p $DB_PORT:$DB_PORT \
    -v  ./db_config/mongo-init.js:/docker-entrypoint-initdb.d/mongo-init.js:ro \
    $MONGODB_IMAGE:$MONGODB_TAG 

# Build and start backend
export BACKEND_NAME="backend"
docker build -t $BACKEND_NAME \
    -f backend/Dockerfile.dev \
    backend

if [ "$(docker ps -q -f name=$BACKEND_NAME)" ]; then
    echo killing container $BACKEND_NAME
    docker kill $BACKEND_NAME
fi

if [ "$(docker ps -a -q -f name=$BACKEND_NAME)" ]; then
    echo removing container $BACKEND_NAME
    docker rm $BACKEND_NAME
fi
docker run -d --name $BACKEND_NAME \
    --network $NETWORK_NAME \
    -e MONGO_DB_HOST=$MONGO_DB_HOST \
    -e DB_NAME=$DB_NAME \
    -e MONGO_DB_USER=$MONGO_DB_USER \
    -e MONGO_DB_PASSWORD=$MONGO_DB_PASSWORD \
    -e DB_NAME=$DB_NAME \
    -e PORT=$DB_PORT \
    -v ./backend/:/app \
    -p $BACKEND_PORT:$BACKEND_PORT \
    backend
    
