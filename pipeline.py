import os
import requests
import tempfile
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from google.cloud import storage
from google.cloud import bigquery
import numpy as np

"""
  ************************
  Devido a limitação na nuvem GCP, devido ao plano Free, não conseguirei demonstrar o upload do arquivo pela pipeline. Desta forma, realizarei o UPLOAD manualmente no bucket
  ************************
  
  # Definição do projeto na GCP
  project_id = "halogen-antenna-426018-b0"
  bucket_name = "halogen-antenna-426018-b0"
  destination_blob_name = "Prata"

  github_file_url = "https://raw.githubusercontent.com/AlexanderAlmeida/pos_graduacao/master/consig.csv"

  def download_and_upload_to_gcs(github_file_url, bucket_name, destination_blob_name, project_id):
    # Download .CSV do GitHub
    response = requests.get(github_file_url)
    if response.status_code == 200:
        csv_data = response.content
    else:
        raise Exception(f"Error downloading file: {response.status_code}")

    # Criação do arquivo temporário para armazenar os dados do download
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(csv_data)
        filename = temp_file.name  # Get the temporary file path

    # Autenticar no Cloud Storage e criar no bucket
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.get_bucket(bucket_name)

    # Upload do .CSV para o Cloud Storage
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(filename)

    # Delete the temporary file after upload
    # (Optional, the file will be automatically deleted when the program exits)
    os.remove(filename)

  download_and_upload_to_gcs(github_file_url, bucket_name, destination_blob_name, project_id)

  print(f"File downloaded from GitHub and uploaded to bucket: {bucket_name}/{destination_blob_name}")

"""


"""
Estaremos utilizando a Arquitetura Medalhão na Pipeline

"""


# Definição do projeto na GCP
project_id = "halogen-antenna-426018-b0"
  
# Definindo a sessão spark
spark = SparkSession.builder.appName("AnaliseCreditoConsignado").getOrCreate()


###########################
# CAMADA BRONZE (Ingestão)
###########################

# Leia o dataset do Cloud Storage
df_fe = spark.read.csv("gs://halogen-antenna-426018-b0/Bronze/consig.csv",header=True, inferSchema=True)
df_fe.show()

# Contando o número de linhas do dataset para observabilidade
num_rows_csv = df_fe.count()
print(f"Number of rows in CSV: {num_rows_csv}")

# Autenticação no BigQuery
cliente_bq = bigquery.Client()

# Defina o nome do projeto e do conjunto de dados no BigQuery
dataset_id = "CREDITO_CONSIGNADO"
 
# Armazenando os dados brutos no bigquery
df_fe.write.format("bigquery").option("temporaryGcsBucket", "").option("writeMethod", "DIRECT").option("project", project_id).option("dataset", dataset_id).option("table", "tbl_consignado_raw").save()

# Verificando a quantidade de registros importados para observabilidade

query = f"SELECT COUNT(*) as total_rows FROM `halogen-antenna-426018-b0.CREDITO_CONSIGNADO.tbl_consignado_raw`"
query_job = cliente_bq.query(query)
result = query_job.result()
num_rows_bq = list(result)[0].total_rows
print(f"Number of rows in BigQuery table: {num_rows_bq}")



#############################
#CAMADA PRATA (Tranformação)
#############################

## Mantenha apenas a primeira ocorrência por CPF
# Verificando duplicidade no CPF, devido a anonimização do campo para este MVP
dataset_sem_duplicatas = df_fe.withColumn("row_num", F.row_number().over(Window.partitionBy("CPF").orderBy(F.col("cpf").desc())))
dataset_sem_duplicatas = dataset_sem_duplicatas.filter(F.col("row_num") == 1)

dataset_sem_duplicatas.show()

dataset_sem_duplicatas = dataset_sem_duplicatas.filter(F.col("row_num") == 1).drop("row_num")
df_fe = dataset_sem_duplicatas


# Crie colunas para cada faixa etária
df_fe = df_fe.withColumn("Faixa_10_20", F.when(df_fe["idade"].between(10, 20), 1).otherwise(0))
df_fe = df_fe.withColumn("Faixa_21_30", F.when(df_fe["idade"].between(21, 30), 1).otherwise(0))
df_fe = df_fe.withColumn("Faixa_31_40", F.when(df_fe["idade"].between(31, 40), 1).otherwise(0))
df_fe = df_fe.withColumn("Faixa_41_50", F.when(df_fe["idade"].between(41, 50), 1).otherwise(0))
df_fe = df_fe.withColumn("Faixa_51_60", F.when(df_fe["idade"].between(51, 60), 1).otherwise(0))
df_fe = df_fe.withColumn("Faixa_61_70", F.when(df_fe["idade"].between(61, 70), 1).otherwise(0))
df_fe = df_fe.withColumn("Faixa_71_80", F.when(df_fe["idade"].between(71, 80), 1).otherwise(0))
df_fe = df_fe.withColumn("Faixa_81_90", F.when(df_fe["idade"].between(81, 90), 1).otherwise(0))

df_fe.show()

# Armazenar a transformação no bucket prata
# Defina o caminho para o arquivo Parquet no bucket
path_parquet = "gs://halogen-antenna-426018-b0/Prata/consig_transform.parquet"

# Salve as novas colunas em Parquet no Cloud Storage
df_fe.write.format("parquet").option("path", path_parquet).save()

#########################
#CAMADA OURO (Carga)
#########################

# Calcule a contagem por faixa etária e UF
df_fe = df_fe.groupBy(["uf"]).agg(F.sum(F.col("margem_saldo")).alias("Margem_Saldo"),F.sum("Faixa_10_20").alias("Faixa_10_20"), F.sum("Faixa_21_30").alias("Faixa_21_30"),F.sum("Faixa_31_40").alias("Faixa_31_40"), F.sum("Faixa_41_50").alias("Faixa_41_50"), F.sum("Faixa_51_60").alias("Faixa_51_60"), F.sum("Faixa_61_70").alias("Faixa_61_70"), F.sum("Faixa_71_80").alias("Faixa_71_80"), F.sum("Faixa_81_90").alias("Faixa_81_90"))
df_fe.show()

# Criação do campo ID
df_fe = df_fe.withColumn('id', F.monotonically_increasing_id())

# Reordenando as colunas e deletando o último campo
df_fe = df_fe.select('id', *df_fe.columns[:-1])

# Exibindo o dataset completo
df_fe.show()

# Salve o dataset processado no BigQuery
df_fe.write.format("bigquery").option("temporaryGcsBucket", "").option("writeMethod", "DIRECT").option("project", project_id).option("dataset", dataset_id).option("table", "tbl_fato").save()



# Parar a sessão SparkSession
spark.stop()