FROM node:18-alpine

WORKDIR /app

# Copiar archivos de dependencias del frontend
COPY frontend/package*.json ./

# Instalar dependencias
RUN npm install

# Copiar código del frontend
COPY frontend/ .

# Exponer puerto
EXPOSE 3000

# Comando por defecto
CMD ["npm", "start"]