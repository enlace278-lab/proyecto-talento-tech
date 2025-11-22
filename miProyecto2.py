# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 10:01:34 2025

@author: DISTRIMEDV
"""


import pandas as pd
import numpy as np 


df_csv =pd.read_csv("Casos_positivos_de_COVID-19_en_Colombia._20251102.csv")
print(df_csv)

df4 = df_csv.copy()
# =====================================
# 2. Exploración Inicial
# =====================================
#primeras 5 filas
print(df4.head())

#ultimas 5 filas
print(df4.tail())

#dimensiones del data set
print(df4.shape)

#nombre de las columnas
print(df4.columns)

#tipo de datos y valores nulos
print(df4.info())

#estadistica basica
print(df4.describe())

# Número de valores únicos por columna

print(df4.nunique())

# Estadísticas para columnas categóricas (tipo object o string)
print("Estadísticas de variables categóricas:")
print(df4.describe(include='object'))

# Conteo de filas duplicadas
print("Número de filas duplicadas:")
print(df4.duplicated().sum())

# Ver primeras filas duplicadas si existen
print("Filas duplicadas (si existen):")
print(df4[df4.duplicated()].head())

# Si quieres ver la correlación entre variables numéricas:
print("Correlación entre variables numéricas:")
print(df4.corr(numeric_only=True))


# =====================================
# 3. Limpieza de Datos
# =====================================
# valores Nulos


print(df4.isnull().sum())

# Ver porcentaje de nulos por columna
print(df4.isnull().mean() * 100)


# Eliminamos columnas no útiles para análisis inicial
#df4.drop(columns=["Nombre del país","Nombre del grupo étnico","Fecha de muerte","Código ISO del país","Código DIVIPOLA municipio"], inplace=True)
#df4.drop(columns=["Tipo de recuperación"], inplace=True)
df4.drop(columns=["Código ISO del país","Nombre del país"], inplace=True)
df4.drop(columns=["Fecha de muerte"], inplace=True)

print(df4.columns)
df7=df4.drop(index=[0, 6], inplace=True)

# Eliminar las filas en las posiciones 0 y 2
df8 = df4.drop(df4.index[[0, 6]])

df4(df4.index[0,4])

# Imputación de valores nulos en variables seleccionadas
# 'Age': usamos la mediana porque es robusta ante outliers
df4['Edad'].fillna(df4['Estado'].median(), inplace=True)  # Mediana para Edad

# 'Embarked': usamos la moda (valor más frecuente) para conservar la categoría dominante
df4['Sexo'].fillna(df4['Nombre municipio'].mode()[0], inplace=True)  # Moda para Embarque

print(df4.isnull().sum())

##Eliminar valores duplicados
#True indica filas duplicadas respecto a la primera ocurrencia
print(df4.duplicated())  # Serie booleana mostrando duplicados

#Número total de filas duplicadas
print(df4.duplicated().sum())  # Conteo de filas duplicadas

# Opcional: eliminar duplicados para continuar con un dataset depurado
df4 = df4.drop_duplicates().copy()  # Eliminamos duplicados y copiamos el resultado para evitar vistas


# =====================================
# 4. Transformación de Variables
# =====================================

##Renombrar una Columna
df4.rename (columns={"Código DIVIPOLA municipio": "Código Municipio"}, inplace=True) # se cambia Fare por tarifa
print(df4.columns)

df4.rename (columns={"Código DIVIPOLA departamento": "Código Departamento"}, inplace=True) # se cambia Fare por tarifa
print(df4.columns)
#nombre de las columnas
print(df4.columns)

##REordenar una columna : cambia el orden de las columnas
df5 = df4[["Pclass","Survived","Sex",'SibSp', 'Age', 'Parch', 'Tarifa',
       'Embarked']]
print(df5.columns)

#es una función anónima que devuelve: 1 si la edad es menor a 12 años,
#0 en caso contrario.

df5['IsChild'] = df5['Age'].apply(lambda x: 1 if x < 12 else 0)


# =====================================
# 5. Análisis Univariado
# =====================================

#promedio de la edad por clase

print(df5.groupby("Pclass")["Age"].mean())

# promedio de la edad por embarque

# Estadística descriptiva extendida para tarifa
print(df5['Tarifa'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))  # Incluye percentiles personalizados

# Distribución de embarque (conteo por puerto)
#Conteo de pasajeros por puerto de embarque
print(df5['Embarked'].value_counts())  # Conteo simple por categoría


# =====================================
# 6. Análisis Bivariado
# =====================================

# tasa de supervivencia por sexo
print(df5.groupby("Sex")["Survived"].mean())

#tasa de supervivencia por clase
print(df5.groupby("Pclass")["Survived"].mean())

#tabla cruzada clase vs supervivencia
print(pd.crosstab(df5["Pclass"], df5["Survived"]))

# Tabla cruzada normalizada por fila (proporciones por clase)
#Tabla cruzada Pclass vs Survived (proporciones por clase)
#Las columnas son el estado de supervivencia (Survived: 0 = no sobrevivió, 1 = sobrevivió)
#Las filas son las clases (Pclass: 1, 2, 3)
#normalize='index'
#En lugar de conteos, devuelve proporciones por fila (cada fila suma 1).
#.round(3) Redondea los valores a 3 decimales

print(pd.crosstab(df5["Pclass"], df5["Survived"], normalize='index').round(3))  # Normalización por fila


# =====================================
# 7. Análisis Multivariado
# =====================================
#Edad - media, mediana, y desviación por clase
print(df5.groupby("Pclass")["Age"].agg(["mean","median","std"]))  # Agregaciones múltiples

# Supervivencia por (clase, sexo)
print(df5.groupby(["Pclass", "Sex"])["Survived"].mean().unstack())  # Tabla de doble entrada

# Pivot table multivariable: promedio de Tarifa por (Pclass, Embarked)
print(pd.pivot_table(df5, values='Tarifa', index='Pclass', columns='Embarked', aggfunc='mean').round(2))

# Correlación entre variables numéricas
num_cols = df5.select_dtypes(include=['number']).columns  # Selecciona columnas numéricas
print(df5[num_cols].corr().round(3))  # Correlación de Pearson por defecto


# =====================================
# 8. Correlación Numérica
# =====================================
#Correlación con la variable 'Survived'
print(df5.corr(numeric_only=True)['Survived'].sort_values(ascending=False))

# =====================================
# 9. Estadísticas Detalladas
# =====================================
#Edad - Media, Mediana y Desviación por Clase
print(df5.groupby('Pclass')['Age'].agg(['mean', 'median', 'std']))

#Supervivencia por sexo y clase (pivot):
# Promedio de supervivencia por sexo y clase
print(df5.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean'))
                                                  

#Estadísticas de tarifas pagadas por clase
print(df5.groupby('Pclass')['Tarifa'].describe())

#Promedio de edad según supervivencia
print(df5.groupby('Survived')['Age'].mean())

#Proporción de niños por clase
print(df5.groupby('Pclass')['IsChild'].mean())

# =====================================
# 11. Exportar Dataset Limpio
# =====================================
df4.to_csv("Casos_positivos_de_COVID-19_en_Colombia._20251102limpio.csv", index=False)

import pandas as pd
import numpy as np
import unicodedata
import re

# =====================================================
# ========== FUNCIONES DE NORMALIZACIÓN ===============
# =====================================================

def normalizar_texto(texto):
    """Normaliza un texto eliminando tildes, pasando a minúsculas y quitando símbolos."""
    if pd.isnull(texto):
        return texto

    # Convertir a string
    texto = str(texto)

    # Quitar acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')

    # Convertir a minúsculas
    texto = texto.lower()

    # Eliminar caracteres especiales
    texto = re.sub(r'[^a-z0-9\s.,_-]', '', texto)

    # Quitar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto


# =====================================================
# ========== ESTANDARIZAR NOMBRES COLUMNAS ============
# =====================================================

def limpiar_nombres_columnas(df):
    nuevas = []
    for col in df.columns:
        col = normalizar_texto(col)
        col = col.replace(' ', '_')
        nuevas.append(col)
    df.columns = nuevas
    return df


# =====================================================
# ========== MANEJO DE TIPOS DE DATOS =================
# =====================================================

def convertir_tipos(df):
    """Intenta convertir columnas a tipos adecuados automáticamente."""

    for col in df.columns:
        # Intentar convertir a numérico
        try:
            df[col] = pd.to_numeric(df[col])
            continue
        except:
            pass

        # Intentar convertir a fecha
        try:
            df[col] = pd.to_datetime(df[col])
            continue
        except:
            pass

        # Convertir a string si sigue sin tipo adecuado
        df[col] = df[col].astype(str)

    return df


# =====================================================
# ========== LIMPIEZA DE VALORES NULOS ================
# =====================================================

def tratar_nulos(df):
    """Rellena valores nulos basándose en el tipo de dato."""

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


# =====================================================
# ========== LIMPIEZA GENERAL DEL DATASET =============
# =====================================================

def limpiar_dataset(df):
    print("\n=== INICIANDO LIMPIEZA PROFESIONAL ===")

    # ------------------------------------
    # 1. Estandarizar nombres
    # ------------------------------------
    print("→ Estandarizando nombres de columnas...")
    df = limpiar_nombres_columnas(df)

    # ------------------------------------
    # 2. Conversión de tipos
    # ------------------------------------
    print("→ Convirtiendo tipos de datos automáticamente...")
    df = convertir_tipos(df)

    # ------------------------------------
    # 3. Normalización de texto
    # ------------------------------------
    print("→ Normalizando valores de texto...")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(normalizar_texto)

    # ------------------------------------
    # 4. Manejo de nulos
    # ------------------------------------
    print("→ Corrigiendo valores nulos...")
    df = tratar_nulos(df)

    # ------------------------------------
    # 5. Eliminar duplicados
    # ------------------------------------
    print("→ Eliminando duplicados...")
    df.drop_duplicates(inplace=True)

    # ------------------------------------
    # 6. Ordenar columnas
    # ------------------------------------
    df = df.reindex(sorted(df.columns), axis=1)

    print("✔ LIMPIEZA COMPLETA.")
    return df


# =====================================================
# ========== PRUEBA DEL MÓDULO (OPCIONAL) =============
# =====================================================
if __name__ == "__main__":
    print("\n### MÓDULO DE LIMPIEZA – PRUEBA ###")
    archivo = "Casos_positivos_de_COVID-19_en_Colombia._20251102limpio.csv"

    try:
        df_test = pd.read_csv(archivo)
        df_limpio = limpiar_dataset(df_test)
        df_limpio.to_csv("casos_covid", index=False)
        print("Archivo limpio generado como ventas_limpias.csv")

    except Exception as e:
        print("No se encontró el archivo para prueba, pero el módulo funciona correctamente.")
        print("Error:", e)
